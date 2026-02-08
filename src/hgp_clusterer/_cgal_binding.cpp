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
#include "kernels_geogram.hpp"

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
};

// ==============================================================================================
// Main Computation Function
// ==============================================================================================

py::tuple compute_delaunay(
    py::array_t<double, py::array::c_style | py::array::forcecast> input_points,
    int K_max,
    std::string precision = "safe",
    bool verbose = false,
    std::string backend = "cgal"
) {
    // 1. Parse Input
    py::buffer_info buf = input_points.request();
    if (buf.ndim != 2) throw std::runtime_error("Input must be 2D array");
    
    size_t N = buf.shape[0];
    size_t dim = buf.shape[1];
    
    if (N < 2) return py::make_tuple(py::array_t<int32_t>(), py::array_t<double>()); // Empty
    if (K_max < 1) return py::make_tuple(py::array_t<int32_t>(), py::array_t<double>());

    // Copy to std::vector because kernels.hpp expects it
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
    std::unique_ptr<WeightedDelaunayTraits> kernel;

    if (backend == "geogram") {
        #ifdef HGP_WITH_GEOGRAM
        kernel = std::make_unique<GeogramDelaunayImpl>();
        if (verbose) std::cout << "[Backend] Using Geogram" << std::endl;
        #else
        throw std::runtime_error("Geogram backend not compiled (HGP_WITH_GEOGRAM not defined).");
        #endif
    } else {
        // Default to CGAL if backend="cgal" or unknown
        if (backend != "cgal" && verbose) {
             std::cout << "[Warning] Unknown backend '" << backend << "', defaulting to CGAL." << std::endl;
        }
        
        #ifdef HGP_WITH_CGAL
        bool exact_mode = (precision == "exact");
        kernel = create_cgal_kernel(dim, exact_mode);
        if (verbose) std::cout << "[Backend] Using CGAL (" << (exact_mode ? "Exact" : "Safe") << ")" << std::endl;
        #else
        throw std::runtime_error("CGAL backend not compiled (HGP_WITH_CGAL not defined).");
        #endif
    }

    if (!kernel) {
        throw std::runtime_error("Failed to initialize kernel (Unsupported dimension/kernel combination)");
    }

    // 2. Initial Step (k=1) -> Standard Delaunay (Weighted with weights=0)
    std::vector<std::vector<int>> prev_simplices; 
    
    {
        // For k=1, we can use the specialized standard Delaunay call which might be faster
        auto edges = kernel->get_standard_delaunay_edges(ptr, N, dim);
        
        // Sort and Unique (Standard procedure)
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
        // Return result (Edges)
        // Shape (M, 2)
        auto result = py::array_t<int32_t>({(long)prev_simplices.size(), (long)2});
        auto r_ptr = result.mutable_unchecked<2>();
        for(size_t i=0; i<prev_simplices.size(); ++i) {
            r_ptr(i, 0) = prev_simplices[i][0];
            r_ptr(i, 1) = prev_simplices[i][1];
        }

        // Compute weights for edges (squared distance / 4)
        auto weights = py::array_t<double>(prev_simplices.size());
        auto w_ptr = weights.mutable_unchecked<1>();
        
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, prev_simplices.size()), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
        #else
        for(size_t i=0; i<prev_simplices.size(); ++i) {
        #endif
                int idx1 = prev_simplices[i][0];
                int idx2 = prev_simplices[i][1];
                double dist_sq = 0.0;
                for(size_t d=0; d<dim; ++d) {
                    double diff = ptr[idx1*dim + d] - ptr[idx2*dim + d];
                    dist_sq += diff * diff;
                }
                w_ptr(i) = dist_sq * 0.25;
        #ifdef CGAL_LINKED_WITH_TBB
            }
        });
        #else
        }
        #endif

        return py::make_tuple(result, weights);
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

                // Radius^2 - Distance^2
                // Order-k definition logic (Lifted weight)
                bary_weights[i] = center_sq_norm - (sum_sq_norms * inv_k);
        #ifdef CGAL_LINKED_WITH_TBB
            }
        });
        #else
        }
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
        #else
        }
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

    // 4. Return Result (Simplices, Weights)
    if (prev_simplices.empty()) {
        return py::make_tuple(py::array_t<int32_t>(), py::array_t<double>());
    }
    
    size_t final_k = prev_simplices[0].size();
    auto simplices_array = py::array_t<int32_t>({(long)prev_simplices.size(), (long)final_k});
    auto s_ptr = simplices_array.mutable_unchecked<2>();
    
    // Fill simplices
    for(size_t i=0; i<prev_simplices.size(); ++i) {
        for(size_t j=0; j<final_k; ++j) {
            s_ptr(i, j) = prev_simplices[i][j];
        }
    }

    // Compute weights (Squared Radii)
    if(verbose) std::cout << "[Info] Computing squared radii for " << prev_simplices.size() << " simplices..." << std::endl;
    
    auto weights_array = py::array_t<double>(prev_simplices.size());
    auto w_ptr = weights_array.mutable_unchecked<1>();
    
    #ifdef CGAL_LINKED_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, prev_simplices.size()), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i=r.begin(); i!=r.end(); ++i) {
    #else
    for(size_t i=0; i<prev_simplices.size(); ++i) {
    #endif
            w_ptr(i) = kernel->compute_simplex_squared_radius(
                ptr, prev_simplices[i], dim
            );
    #ifdef CGAL_LINKED_WITH_TBB
        }
    });
    #else
    }
    #endif
    
    return py::make_tuple(simplices_array, weights_array);
}

PYBIND11_MODULE(cgal_binding, m) {
    m.doc() = "CGAL/Geogram-based Order-K Delaunay Triangulation Binding";
    m.def("compute_delaunay", &compute_delaunay, "Compute Order-K Delaunay",
          py::arg("points"), py::arg("K_max"), py::arg("precision")="safe", py::arg("verbose")=false, py::arg("backend")="geogram");
}