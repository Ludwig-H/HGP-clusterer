#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <thread>
#include <chrono>

// Include the existing kernels logic
// We assume kernels.hpp is in the include path
#include "kernels.hpp"

// TBB for parallelism
#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
#else
#define tbb_par_for(loop) loop
#endif

namespace py = pybind11;

// ==============================================================================================
// Utilities
// ==============================================================================================

double norm_sq(const double* p, size_t dim) {
    double sum = 0;
    for(size_t i=0; i<dim; ++i) sum += p[i]*p[i];
    return sum;
}

struct PointCloud {
    const double* data_ptr; 
    std::vector<double> sq_norms; 
    size_t dim;
    size_t N;

    PointCloud(const double* ptr, size_t n_points, size_t dimensions) 
        : data_ptr(ptr), dim(dimensions), N(n_points) {
        
        sq_norms.resize(N);
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
                sq_norms[i] = norm_sq(&data_ptr[i*dim], dim);
            }
        });
        #else
        for(size_t i=0; i<N; ++i) sq_norms[i] = norm_sq(&data_ptr[i*dim], dim);
        #endif
    }
    
    // Helper to get data vector for CGAL kernels which expect std::vector
    // This is a slight inefficiency: existing kernels.hpp expects std::vector<double>
    // We might need to copy if we can't adapt kernels.hpp easily.
    // Looking at kernels.hpp: get_finite_edges takes const std::vector<double>& flat_points.
    // So we MUST copy the input numpy array to a vector if we don't change kernels.hpp.
    // For now, to be safe and fast on dev time, we will copy.
    // OPTIMIZATION LATER: Adapt kernels.hpp to take pointers/spans.
};

// ==============================================================================================
// Main Computation Function
// ==============================================================================================

py::array_t<int32_t> compute_delaunay(
    py::array_t<double, py::array::c_style | py::array::forcecast> input_points,
    int K_max,
    std::string precision = "safe",
    bool verbose = false
) {
    // 1. Parse Input
    py::buffer_info buf = input_points.request();
    if (buf.ndim != 2) throw std::runtime_error("Input must be 2D array");
    
    size_t N = buf.shape[0];
    size_t dim = buf.shape[1];
    
    if (N < 2) return py::array_t<int32_t>(); // Empty
    if (K_max < 1) return py::array_t<int32_t>();

    // Copy to std::vector because kernels.hpp expects it
    // TODO: Optimize this copy out by modifying kernels.hpp to use spans
    const double* ptr = static_cast<double*>(buf.ptr);
    
    PointCloud cloud(ptr, N, dim); // Recalculate sq_norms

    // Setup TBB
    #ifdef CGAL_LINKED_WITH_TBB
    int nthreads = std::thread::hardware_concurrency();
    if(const char* env = std::getenv("CGAL_NTHREADS")) nthreads = std::atoi(env);
    // Limit global parallelism
    static tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nthreads);
    #endif

    // Setup Kernel
    bool exact_mode = (precision == "exact");
    auto kernel = create_kernel(dim, exact_mode);
    if (!kernel) {
        throw std::runtime_error("Unsupported dimension/kernel combination");
    }

    // 2. Initial Step (k=1) -> Standard Delaunay (Weighted with weights=0)
    std::vector<std::vector<int>> prev_simplices; 
    
    {
        std::vector<double> zero_weights(N, 0.0);
        auto edges = kernel->get_finite_edges(ptr, zero_weights, N, dim);
        
        // Sort and Unique
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_sort(edges.begin(), edges.end(), [](const auto& a, const auto& b){
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });
        #else
        std::sort(edges.begin(), edges.end());
        #endif
        auto last = std::unique(edges.begin(), edges.end());
        edges.erase(last, edges.end());

        prev_simplices.reserve(edges.size());
        for(auto& p : edges) {
            if(p.first < p.second) prev_simplices.push_back({p.first, p.second});
            else prev_simplices.push_back({p.second, p.first});
        }
        
        if (verbose) std::cout << "[Step 1] Found " << prev_simplices.size() << " edges.\n";
    }

    if (K_max == 1) {
        // Return result
        // Shape (M, 2)
        auto result = py::array_t<int32_t>({(long)prev_simplices.size(), (long)2});
        auto r_ptr = result.mutable_unchecked<2>();
        for(size_t i=0; i<prev_simplices.size(); ++i) {
            r_ptr(i, 0) = prev_simplices[i][0];
            r_ptr(i, 1) = prev_simplices[i][1];
        }
        return result;
    }

    // 3. Iterative Loop (k=2 to K)
    for (int k = 2; k <= K_max; ++k) {
        size_t n_prev = prev_simplices.size();
        if (n_prev == 0) break;

        // A. Compute Barycenters & Weights
        std::vector<double> bary_coords(n_prev * dim);
        std::vector<double> bary_weights(n_prev);

        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_prev), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
        #else
        for(size_t i=0; i<n_prev; ++i) {
        #endif
                const auto& simp = prev_simplices[i];
                double inv_k = 1.0 / double(simp.size());
                
                double sum_sq_norms = 0.0;
                double center_sq_norm = 0.0;
                std::vector<double> center(dim, 0.0);

                for(int idx : simp) {
                    sum_sq_norms += cloud.sq_norms[idx];
                    for(size_t d=0; d<dim; ++d) {
                        center[d] += ptr[idx * dim + d];
                    }
                }

                for(size_t d=0; d<dim; ++d) {
                    center[d] *= inv_k;
                    bary_coords[i * dim + d] = center[d];
                    center_sq_norm += center[d] * center[d];
                }

                bary_weights[i] = center_sq_norm - (sum_sq_norms * inv_k);
        #ifdef CGAL_LINKED_WITH_TBB
            }
        });
        #endif

        // B. Weighted Delaunay on Barycenters
        auto dual_edges = kernel->get_finite_edges(bary_coords, bary_weights, n_prev, dim);

        if (dual_edges.empty()) {
            prev_simplices.clear();
            break;
        }

        // C. Reconstitution / Union
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::concurrent_vector<std::vector<int>> candidates;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, dual_edges.size()), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
        #else
        std::vector<std::vector<int>> candidates;
        for(size_t i=0; i<dual_edges.size(); ++i) {
        #endif
                int idx_a = dual_edges[i].first;
                int idx_b = dual_edges[i].second;
                
                const auto& sA = prev_simplices[idx_a];
                const auto& sB = prev_simplices[idx_b];
                
                std::vector<int> merged; 
                merged.reserve(sA.size() + 1);
                
                size_t ia = 0, ib = 0;
                while(ia < sA.size() && ib < sB.size()) {
                    if(sA[ia] < sB[ib]) merged.push_back(sA[ia++]);
                    else if(sB[ib] < sA[ia]) merged.push_back(sB[ib++]);
                    else { // equal
                        merged.push_back(sA[ia]);
                        ia++; ib++;
                    }
                }
                while(ia < sA.size()) merged.push_back(sA[ia++]);
                while(ib < sB.size()) merged.push_back(sB[ib++]);
                
                if (merged.size() == sA.size() + 1) {
                    candidates.push_back(merged);
                }
        #ifdef CGAL_LINKED_WITH_TBB
            }
        });
        #endif

        if (candidates.empty()) {
            prev_simplices.clear();
            break;
        }

        std::vector<std::vector<int>> next_simplices;
        next_simplices.reserve(candidates.size());
        for(auto& c : candidates) next_simplices.push_back(std::move(c));

        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_sort(next_simplices.begin(), next_simplices.end());
        #else
        std::sort(next_simplices.begin(), next_simplices.end());
        #endif

        auto last_unique = std::unique(next_simplices.begin(), next_simplices.end());
        next_simplices.erase(last_unique, next_simplices.end());

        prev_simplices = std::move(next_simplices);
        
        if(verbose) std::cout << "[Step " << k << "] Generated " << prev_simplices.size() << " simplices\n";
    }

    // 4. Return Result
    if (prev_simplices.empty()) return py::array_t<int32_t>();
    
    size_t final_k = prev_simplices[0].size();
    auto result = py::array_t<int32_t>({(long)prev_simplices.size(), (long)final_k});
    auto r_ptr = result.mutable_unchecked<2>();
    
    for(size_t i=0; i<prev_simplices.size(); ++i) {
        for(size_t j=0; j<final_k; ++j) {
            r_ptr(i, j) = prev_simplices[i][j];
        }
    }
    
    return result;
}

PYBIND11_MODULE(cgal_binding, m) {
    m.doc() = "CGAL-based Order-K Delaunay Triangulation Binding";
    m.def("compute_delaunay", &compute_delaunay, "Compute Order-K Delaunay",
          py::arg("points"), py::arg("K_max"), py::arg("precision")="safe", py::arg("verbose")=false);
}
