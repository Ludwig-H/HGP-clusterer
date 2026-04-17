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
#include <numeric>

// Include the existing kernels logic
// We assume kernels.hpp is in the include path
#include "kernels.hpp"
#include "kernels_geogram.hpp"
#ifdef HGP_WITH_GEOGRAM
#include <geogram/basic/process.h>
#endif

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

    // Setup Parallelism (TBB & Geogram)
    int nthreads = std::thread::hardware_concurrency();
    
    // Check Environment Variables
    if(const char* env = std::getenv("GEOGRAM_NUM_THREADS")) {
        nthreads = std::atoi(env);
    } else if(const char* env = std::getenv("CGAL_NTHREADS")) {
        nthreads = std::atoi(env);
    }
    
    // Sanity check
    if (nthreads < 1) nthreads = 1;

    #ifdef CGAL_LINKED_WITH_TBB
    // Limit TBB global parallelism to match Geogram's allocation
    // This prevents "thread storms" where TBB and Geogram fight for resources
    static tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nthreads);
    #endif

    // Setup Kernel
    std::unique_ptr<WeightedDelaunayTraits> kernel;

    if (backend == "geogram") {
        #ifdef HGP_WITH_GEOGRAM
        kernel = std::make_unique<GeogramDelaunayImpl>();
        if (verbose) std::cout << "[Backend] Using Geogram" << std::endl;

        // Configure Geogram Threads (commented out due to undefined symbol in some libgeogram builds)
        // GEO::Process::set_max_threads(nthreads);
        // if (verbose) std::cout << "[Geogram] Max threads set to: " << nthreads << std::endl;

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
    // Flattened simplices: [s1_v1, s1_v2, s2_v1, s2_v2, ...]
    std::vector<int> prev_simplices; 
    size_t current_k = 2; // Simplex size for K=1 (Edge = 2 vertices)
    
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

        prev_simplices.reserve(edges.size() * 2);
        for(auto& p : edges) {
            if(p.first < p.second) {
                prev_simplices.push_back(p.first);
                prev_simplices.push_back(p.second);
            } else {
                prev_simplices.push_back(p.second);
                prev_simplices.push_back(p.first);
            }
        }
        
        if (verbose) std::cout << "[Step 1] Found " << edges.size() << " edges.\n";
    }

    if (K_max == 1) {
        // Return result (Edges)
        // Shape (M, 2)
        size_t n_simplices = prev_simplices.size() / 2;
        auto result = py::array_t<int32_t>({(long)n_simplices, (long)2});
        auto r_ptr = result.mutable_unchecked<2>();
        
        // Compute weights for edges (squared distance / 4)
        auto weights = py::array_t<double>(n_simplices);
        auto w_ptr = weights.mutable_unchecked<1>();
        
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_simplices), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
        #else
        for(size_t i=0; i<n_simplices; ++i) {
        #endif
                int idx1 = prev_simplices[i*2];
                int idx2 = prev_simplices[i*2 + 1];
                
                r_ptr(i, 0) = idx1;
                r_ptr(i, 1) = idx2;

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
        size_t n_prev = prev_simplices.size() / current_k;
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
                double inv_k = 1.0 / double(current_k);
                double sum_sq_norms = 0.0;
                double center_sq_norm = 0.0;
                
                // Temp center buffer not strictly needed if we iterate dim
                
                // Direct computation to avoid vector allocation
                const int* s_ptr = &prev_simplices[i * current_k];
                
                for(size_t d=0; d<dim; ++d) {
                    double coord_sum = 0.0;
                    for(size_t v=0; v<current_k; ++v) {
                        coord_sum += ptr[s_ptr[v] * dim + d];
                    }
                    double c_val = coord_sum * inv_k;
                    bary_coords[i * dim + d] = c_val;
                    center_sq_norm += c_val * c_val;
                }
                
                for(size_t v=0; v<current_k; ++v) {
                    sum_sq_norms += cloud.sq_norms[s_ptr[v]];
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
        size_t next_k = current_k + 1;
        
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::concurrent_vector<int> candidates_flat;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, dual_edges.size()), [&](const tbb::blocked_range<size_t>& r) {
            // Thread-local buffer to reduce concurrent overhead
            std::vector<int> local_buf; 
            local_buf.reserve((r.end() - r.begin()) * next_k); // Heuristic
            
            for(size_t i=r.begin(); i!=r.end(); ++i) {
        #else
        std::vector<int> candidates_flat;
        candidates_flat.reserve(dual_edges.size() * next_k);
        for(size_t i=0; i<dual_edges.size(); ++i) {
            std::vector<int>& local_buf = candidates_flat; // Alias for non-tbb
        #endif
                int idx_a = dual_edges[i].first;
                int idx_b = dual_edges[i].second;
                
                const int* sA = &prev_simplices[idx_a * current_k];
                const int* sB = &prev_simplices[idx_b * current_k];
                
                // Merge sorted arrays
                // We write directly to a small stack buffer then push to vector
                // But since current_k is small, we can just do it.
                // However, we need to know if size is exactly k+1
                
                int merged[256]; // Should be enough for k < 255
                // If k is huge, this stack alloc might be risky, but usually k < 50
                // Fallback to heap if needed? No, order-k usually < 20.
                
                size_t ia = 0, ib = 0, im = 0;
                while(ia < current_k && ib < current_k) {
                    if(sA[ia] < sB[ib]) merged[im++] = sA[ia++];
                    else if(sB[ib] < sA[ia]) merged[im++] = sB[ib++];
                    else { // equal
                        merged[im++] = sA[ia];
                        ia++; ib++;
                    }
                }
                while(ia < current_k) merged[im++] = sA[ia++];
                while(ib < current_k) merged[im++] = sB[ib++];
                
                if (im == next_k) {
                    for(size_t x=0; x<next_k; ++x) local_buf.push_back(merged[x]);
                }
        #ifdef CGAL_LINKED_WITH_TBB
            }
            // Flush local to concurrent
            if(!local_buf.empty()) {
                auto range = candidates_flat.grow_by(local_buf.size());
                std::copy(local_buf.begin(), local_buf.end(), range);
            }
        });
        #else
        }
        #endif

        if (candidates_flat.empty()) {
            prev_simplices.clear();
            break;
        }

        // Convert to contiguous std::vector if using TBB, because concurrent_vector is segmented
        // and we need contiguous memory for pointer arithmetic in sort/unique logic.
        #ifdef CGAL_LINKED_WITH_TBB
        std::vector<int> candidates_contiguous(candidates_flat.begin(), candidates_flat.end());
        #else
        std::vector<int>& candidates_contiguous = candidates_flat;
        #endif

        // D. Sort and Unique (Indirect)
        size_t n_candidates = candidates_contiguous.size() / next_k;
        std::vector<size_t> indices(n_candidates);
        std::iota(indices.begin(), indices.end(), 0);

        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            const int* a = &candidates_contiguous[i * next_k];
            const int* b = &candidates_contiguous[j * next_k];
            for(size_t x=0; x<next_k; ++x) {
                if (a[x] != b[x]) return a[x] < b[x];
            }
            return false;
        });
        #else
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            const int* a = &candidates_contiguous[i * next_k];
            const int* b = &candidates_contiguous[j * next_k];
            for(size_t x=0; x<next_k; ++x) {
                if (a[x] != b[x]) return a[x] < b[x];
            }
            return false;
        });
        #endif

        // Unique copy
        std::vector<int> next_simplices_flat;
        next_simplices_flat.reserve(candidates_contiguous.size()); // Max size
        
        if (n_candidates > 0) {
            // Always push first
            const int* first = &candidates_contiguous[indices[0] * next_k];
            next_simplices_flat.insert(next_simplices_flat.end(), first, first + next_k);
            
            for(size_t i=1; i<n_candidates; ++i) {
                const int* curr = &candidates_contiguous[indices[i] * next_k];
                const int* prev = &candidates_contiguous[indices[i-1] * next_k];
                
                bool diff = false;
                for(size_t x=0; x<next_k; ++x) {
                    if (curr[x] != prev[x]) {
                        diff = true;
                        break;
                    }
                }
                
                if(diff) {
                    next_simplices_flat.insert(next_simplices_flat.end(), curr, curr + next_k);
                }
            }
        }

        prev_simplices = std::move(next_simplices_flat);
        current_k = next_k; // Update k size
        
        if(verbose) std::cout << "[Step " << k << "] Generated " << prev_simplices.size() / current_k << " simplices\n";
    }

    // 4. Return Result (Simplices, Weights)
    size_t n_final_simplices = prev_simplices.empty() ? 0 : prev_simplices.size() / current_k;
    
    if (n_final_simplices == 0) {
        return py::make_tuple(py::array_t<int32_t>(), py::array_t<double>());
    }
    
    auto simplices_array = py::array_t<int32_t>({(long)n_final_simplices, (long)current_k});
    auto s_ptr = simplices_array.mutable_unchecked<2>();
    
    // Fill simplices
    #ifdef CGAL_LINKED_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_final_simplices), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i=r.begin(); i!=r.end(); ++i) {
            const int* src = &prev_simplices[i * current_k];
            for(size_t j=0; j<current_k; ++j) {
                s_ptr(i, j) = src[j];
            }
        }
    });
    #else
    for(size_t i=0; i<n_final_simplices; ++i) {
        const int* src = &prev_simplices[i * current_k];
        for(size_t j=0; j<current_k; ++j) {
            s_ptr(i, j) = src[j];
        }
    }
    #endif

    // Compute weights (Squared Radii)
    if(verbose) std::cout << "[Info] Computing squared radii for " << n_final_simplices << " simplices..." << std::endl;
    
    auto weights_array = py::array_t<double>(n_final_simplices);
    auto w_ptr = weights_array.mutable_unchecked<1>();
    
    #ifdef CGAL_LINKED_WITH_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_final_simplices), [&](const tbb::blocked_range<size_t>& r) {
        for(size_t i=r.begin(); i!=r.end(); ++i) {
    #else
    for(size_t i=0; i<n_final_simplices; ++i) {
    #endif
            w_ptr(i) = kernel->compute_simplex_squared_radius(
                ptr, &prev_simplices[i * current_k], current_k, dim
            );
    #ifdef CGAL_LINKED_WITH_TBB
        }
    });
    #else
    }
    #endif
    
    return py::make_tuple(simplices_array, weights_array);
}

PYBIND11_MODULE(geometry_binding, m) {
    m.doc() = "Geometry binding (CGAL/Geogram) for HGP";
    m.def("compute_delaunay", &compute_delaunay, "Compute Order-K Delaunay",
          py::arg("points"), py::arg("K_max"), py::arg("precision")="safe", py::arg("verbose")=false, py::arg("backend")="geogram");
}
