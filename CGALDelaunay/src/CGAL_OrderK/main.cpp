#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <thread>
#include <chrono>

#include "npy.hpp"
#include "kernels.hpp"

// TBB for parallelism
#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/concurrent_vector.h>
#else
// Fallback if TBB is missing (should not happen based on plan, but good for safety)
#define tbb_par_for(loop) loop
#endif

// ==============================================================================================
// Utilities
// ==============================================================================================

// Compute squared Euclidean norm of a point
double norm_sq(const double* p, size_t dim) {
    double sum = 0;
    for(size_t i=0; i<dim; ++i) sum += p[i]*p[i];
    return sum;
}

// Compute barycenter and weight for a simplex
// barycenter = sum(points) / k
// weight = ||barycenter||^2 - (sum(||p||^2) / k)
// Actually, formula in prompt: w = ||c||^2 - 1/k * sum(||p||^2)
// This corresponds to the power radius of the minimal enclosing sphere of a regular simplex?
// Let's stick to the prompt's formula rigorously.
struct BarycenterInfo {
    std::vector<double> center;
    double weight;
};

// Flattened storage for points to avoid pointer chasing
struct PointCloud {
    std::vector<double> data; // size = N * dim
    std::vector<double> sq_norms; // size = N
    size_t dim;
    size_t N;

    PointCloud(std::vector<double>&& d, size_t dimensions) : data(std::move(d)), dim(dimensions) {
        N = data.size() / dim;
        sq_norms.resize(N);
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, N), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
                sq_norms[i] = norm_sq(&data[i*dim], dim);
            }
        });
        #else
        for(size_t i=0; i<N; ++i) sq_norms[i] = norm_sq(&data[i*dim], dim);
        #endif
    }
};

// ==============================================================================================
// Core Logic
// ==============================================================================================

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " input.npy K output.npy [precision=safe/exact] [verbose=0/1]\n";
        return 1;
    }

    std::string input_path = argv[1];
    int K_max = std::atoi(argv[2]);
    std::string output_path = argv[3];
    std::string precision = (argc > 4) ? argv[4] : "safe";
    bool verbose = (argc > 5) ? (std::atoi(argv[5]) > 0) : false;

    if (K_max < 1) return 0;

    // 1. Load Data
    std::vector<double> raw_points;
    size_t rows=0, cols=0;
    if (!NpyIO::load_npy_double(input_path.c_str(), raw_points, rows, cols)) {
        std::cerr << "Failed to load input npy.\n";
        return 1;
    }

    if (rows < 2) {
        // Empty result
        NpyIO::save_npy_integers<int>(output_path.c_str(), {}, 0);
        return 0;
    }

    PointCloud cloud(std::move(raw_points), cols);
    if (verbose) std::cerr << "[Info] Loaded " << cloud.N << " points of dimension " << cloud.dim << "\n";

    // Setup TBB
    #ifdef CGAL_LINKED_WITH_TBB
    int nthreads = std::thread::hardware_concurrency();
    if(const char* env = std::getenv("CGAL_NTHREADS")) nthreads = std::atoi(env);
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nthreads);
    if(verbose) std::cerr << "[Info] Using " << nthreads << " threads.\n";
    #endif

    // Setup Kernel
    bool exact_mode = (precision == "exact");
    auto kernel = create_kernel(cloud.dim, exact_mode);
    if (!kernel) {
        std::cerr << "Error: unsupported dimension/kernel combination.\n";
        return 1;
    }

    // 2. Initial Step (k=1) -> Standard Delaunay (Weighted with weights=0 OR Standard)
    // Prompt says: "Initialisation (k=1): Call edges_from_weighted_delaunay... standard Delaunay".
    // Standard Delaunay is Weighted Delaunay with all weights = 0 (or equal).
    // Let's use weights = 0.
    std::vector<std::vector<int>> prev_simplices; // List of simplices (each is vector<int>)
    
    {
        std::vector<double> zero_weights(cloud.N, 0.0);
        auto edges = kernel->get_finite_edges(cloud.data, zero_weights, cloud.N, cloud.dim);
        
        // Convert edges to simplices (k=1, so simplices are pairs of indices)
        // Wait. "k=1" usually means edges?
        // Prompt says: "If K=1, return simply these edges."
        // So for K=1, the "simplices" are indeed the edges of the Delaunay triangulation.
        // Let's standardize: simplex of order k has size k+1.
        // k=1 -> size 2 (edges).
        
        // We need to sort and unique them just in case kernel returned dupes
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
        
        if (verbose) std::cerr << "[Step 1] Found " << prev_simplices.size() << " edges.\n";
    }

    if (K_max == 1) {
        NpyIO::save_npy_integers(output_path.c_str(), prev_simplices, 2);
        return 0;
    }

    // 3. Iterative Loop (k=2 to K)
    for (int k = 2; k <= K_max; ++k) {
        size_t n_prev = prev_simplices.size();
        if (n_prev == 0) break;

        // A. Compute Barycenters & Weights
        // Input: prev_simplices (size k each, wait... previous step was order k-1, so size is k)
        // Output: flat_barycenters (N*dim), weights (N)
        
        std::vector<double> bary_coords(n_prev * cloud.dim);
        std::vector<double> bary_weights(n_prev);

        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_prev), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
        #else
        for(size_t i=0; i<n_prev; ++i) {
        #endif
                const auto& simp = prev_simplices[i];
                double inv_k = 1.0 / double(simp.size());
                
                // Calculate center
                // center = sum(p) / size
                // weight = ||center||^2 - (sum(||p||^2) / size)
                
                double sum_sq_norms = 0.0;
                double center_sq_norm = 0.0;
                
                // Temporary center storage for this thread
                std::vector<double> center(cloud.dim, 0.0);

                for(int idx : simp) {
                    sum_sq_norms += cloud.sq_norms[idx];
                    for(size_t d=0; d<cloud.dim; ++d) {
                        center[d] += cloud.data[idx * cloud.dim + d];
                    }
                }

                for(size_t d=0; d<cloud.dim; ++d) {
                    center[d] *= inv_k;
                    bary_coords[i * cloud.dim + d] = center[d];
                    center_sq_norm += center[d] * center[d];
                }

                bary_weights[i] = center_sq_norm - (sum_sq_norms * inv_k);
        #ifdef CGAL_LINKED_WITH_TBB
            }
        });
        #endif

        // B. Weighted Delaunay on Barycenters
        auto dual_edges = kernel->get_finite_edges(bary_coords, bary_weights, n_prev, cloud.dim);

        if (dual_edges.empty()) {
            prev_simplices.clear();
            break;
        }

        // C. Reconstitution / Union
        // We have edges between indices of `prev_simplices`.
        // Edge (u, v) means we try to merge prev_simplices[u] and prev_simplices[v].
        // Condition: They must share exactly k-1 vertices (size of prev_simplex is k).
        // Union size must be k+1.
        
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
                
                // Access read-only shared data
                const auto& sA = prev_simplices[idx_a];
                const auto& sB = prev_simplices[idx_b];
                
                // Union logic.
                // Since sA and sB are sorted (we ensure sorting at end of loop), 
                // we can use set_intersection or just simple merge scan.
                // Optimization: We know size is k. Intersection must be k-1. Union must be k+1.
                // This implies they differ by exactly 1 element each.
                
                // Let's compute union directly.
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

        // D. Filter and Update
        // Sort and Unique candidates
        if (candidates.empty()) {
            prev_simplices.clear();
            break;
        }

        // Move to std::vector for sorting
        std::vector<std::vector<int>> next_simplices;
        next_simplices.reserve(candidates.size());
        for(auto& c : candidates) next_simplices.push_back(std::move(c)); // TBB copy/move

        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_sort(next_simplices.begin(), next_simplices.end());
        #else
        std::sort(next_simplices.begin(), next_simplices.end());
        #endif

        auto last_unique = std::unique(next_simplices.begin(), next_simplices.end());
        next_simplices.erase(last_unique, next_simplices.end());

        prev_simplices = std::move(next_simplices);
        
        if(verbose) std::cerr << "[Step " << k << "] Generated " << prev_simplices.size() << " simplices of order " << k << " (size " << k+1 << ")\n";
    }

    // 4. Save Result
    size_t final_k = (prev_simplices.empty()) ? 0 : prev_simplices[0].size();
    NpyIO::save_npy_integers(output_path.c_str(), prev_simplices, final_k);

    return 0;
}
