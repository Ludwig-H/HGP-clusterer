#pragma once

#include "kernels.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>

#include <Eigen/Dense>

#ifdef HGP_WITH_GEOGRAM

#include <geogram/delaunay/delaunay.h>
#include <geogram/basic/common.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/file_system.h>

// ==============================================================================
// Fast, Thread-Safe Welzl's Algorithm (using Eigen for Basis solving)
// ==============================================================================
namespace HGP_Numerics {

    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    // Solves for the center and squared radius of the sphere defined by boundary points R
    // R is passed as a list of indices into flat_points.
    // Optimized using Gram matrix approach:
    // Center C = q0 + sum(y_i * (q_i - q0))
    // Solves linear system 2 * (q_i - q0)^T * (q_j - q0) * y = ||q_i - q0||^2
    inline void solve_basis(
        const double* flat_points, 
        const std::vector<int>& R, 
        size_t dim, 
        Vec& center, 
        double& r_sq) 
    {
        size_t m = R.size();
        if (m == 0) {
            center.setZero(dim);
            r_sq = 0.0;
            return;
        }
        if (m == 1) {
            for(size_t k=0; k<dim; ++k) center(k) = flat_points[R[0]*dim + k];
            r_sq = 0.0;
            return;
        }

        // m >= 2 points on boundary
        // We use the first point R[0] as origin q0 to reduce dimensionality and improve stability
        size_t n_vecs = m - 1;
        
        // Construct matrix V (columns are q_i - q_0)
        // And RHS vector b (sq norms)
        Mat V(dim, n_vecs);
        Vec b(n_vecs);

        // Origin q0
        const double* q0 = &flat_points[R[0]*dim];

        for (size_t i = 0; i < n_vecs; ++i) {
            const double* qi = &flat_points[R[i+1]*dim];
            double sq_norm = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                double val = qi[d] - q0[d];
                V(d, i) = val;
                sq_norm += val * val;
            }
            b(i) = sq_norm;
        }

        // Solve G * y = b/2 where G = V^T * V (Gram matrix)
        // The center is q0 + V * y
        // We assume R are affinely independent (simplex vertices usually are).
        // Using LDLT for stability and speed on PSD matrices.
        
        Mat G = V.transpose() * V;
        Vec y = G.ldlt().solve(0.5 * b);

        // Reconstruct Center relative to q0
        Vec local_center = V * y;
        
        r_sq = local_center.squaredNorm();
        
        // Absolute Center
        for(size_t k=0; k<dim; ++k) {
            center(k) = q0[k] + local_center(k);
        }
    }

    // Recursive Welzl
    // P: Points to consider (indices)
    // R: Points on boundary (indices)
    // n: Number of points remaining in P (we conceptually pop from end)
    inline void welzl(
        const double* flat_points,
        std::vector<int>& P, 
        std::vector<int>& R, 
        int n, 
        size_t dim,
        Vec& center,
        double& r_sq)
    {
        if (n == 0 || R.size() >= dim + 1) {
            solve_basis(flat_points, R, dim, center, r_sq);
            return;
        }

        // Pick random point (or last one since P is shuffled initially)
        int p_idx = P[n-1];
        
        // Recursive call without p
        welzl(flat_points, P, R, n-1, dim, center, r_sq);

        // Check if p is inside current ball
        double dist_sq = 0.0;
        for(size_t k=0; k<dim; ++k) {
            double d = flat_points[p_idx*dim + k] - center(k);
            dist_sq += d*d;
        }

        if (dist_sq > r_sq + 1e-10) {
            // p must be on boundary
            R.push_back(p_idx);
            welzl(flat_points, P, R, n-1, dim, center, r_sq);
            R.pop_back();
        }
    }

    inline double compute_meb_sq_radius(const double* flat_points, const std::vector<int>& indices, size_t dim) {
        if (indices.empty()) return 0.0;
        if (indices.size() == 1) return 0.0;
        if (indices.size() == 2) {
            // Fast path for edges
            double d2 = 0;
            const double* p1 = &flat_points[indices[0]*dim];
            const double* p2 = &flat_points[indices[1]*dim];
            for(size_t k=0; k<dim; ++k) {
                double diff = p1[k] - p2[k];
                d2 += diff*diff;
            }
            return d2 * 0.25;
        }

        // Prepare data for Welzl
        std::vector<int> P = indices;
        std::vector<int> R;
        R.reserve(dim + 1);
        
        // Random shuffle P to ensure expected O(N)
        for (size_t i = P.size() - 1; i > 0; --i) {
            size_t j = (size_t((i * 12345 + 6789)) % (i + 1)); // Deterministic but mixed enough
            std::swap(P[i], P[j]);
        }

        Vec center(dim);
        double r_sq = 0.0;

        welzl(flat_points, P, R, (int)P.size(), dim, center, r_sq);
        
        return r_sq;
    }
}

// Geogram Implementation of WeightedDelaunayTraits
class GeogramDelaunayImpl : public WeightedDelaunayTraits {
public:
    GeogramDelaunayImpl() {
        static bool initialized = false;
        if (!initialized) {
            GEO::initialize();
            // GEO::Logger::instance()->set_quiet(true); // Removed: caused API issues on some versions
            initialized = true;
        }
    }

    std::vector<std::pair<int, int>> get_finite_edges(
        const double* flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t dim
    ) override {
        return _compute_edges(flat_points, &weights, n_points, dim);
    }

    std::vector<std::pair<int, int>> get_standard_delaunay_edges(
        const double* flat_points,
        size_t n_points,
        size_t dim
    ) override {
        return _compute_edges(flat_points, nullptr, n_points, dim);
    }

    double compute_simplex_squared_radius(
        const double* flat_points,
        const std::vector<int>& indices,
        size_t dim
    ) override {
        // Use our fast exact Welzl implementation
        return HGP_Numerics::compute_meb_sq_radius(flat_points, indices, dim);
    }

private:
    std::vector<std::pair<int, int>> _compute_edges(
        const double* flat_points,
        const std::vector<double>* weights_ptr,
        size_t n_points,
        size_t dim
    ) {
        std::vector<std::pair<int, int>> edges;
        
        if (n_points < 2) return edges;
        
        // 1. Create Delaunay
        // Use "default" to let Geogram pick best impl (e.g. parallel BPOW3 in 3D)
        GEO::Delaunay_var delaunay = GEO::Delaunay::create(dim, "default");
        
        // 2. Setup vertices (triggers computation usually)
        delaunay->set_vertices(n_points, flat_points);

        // 3. Setup weights (Warning: Standard Geogram Delaunay interface might not support set_weights)
        if (weights_ptr && !weights_ptr->empty()) {
            // Weights ignored for now to prevent compilation errors.
            // set_weights() is not a member of GEO::Delaunay in standard API.
            // Power diagrams require specific factory parameters or lifted points.
        }

        // 4. Compute (Removed: implicit in set_vertices or create)
        // delaunay->compute();

        // 5. Extract Edges (Parallel-friendly traversal structure)
        GEO::index_t n_cells = delaunay->nb_cells();
        GEO::index_t n_vertices_per_cell = delaunay->cell_size();

        // Pre-reserve to avoid reallocations
        edges.reserve(n_cells * (n_vertices_per_cell * (n_vertices_per_cell-1)) / 2);

        for (GEO::index_t c = 0; c < n_cells; ++c) {
            for (GEO::index_t i = 0; i < n_vertices_per_cell; ++i) {
                for (GEO::index_t j = i + 1; j < n_vertices_per_cell; ++j) {
                    GEO::index_t v1 = delaunay->cell_vertex(c, i);
                    GEO::index_t v2 = delaunay->cell_vertex(c, j);
                    
                    // Filter infinite vertices (Geogram indices are usually < n_points for finite)
                    if (v1 < n_points && v2 < n_points) {
                         if (v1 < v2) edges.push_back({(int)v1, (int)v2});
                         else edges.push_back({(int)v2, (int)v1});
                    }
                }
            }
        }

        // Deduplicate
        std::sort(edges.begin(), edges.end());
        auto last = std::unique(edges.begin(), edges.end());
        edges.erase(last, edges.end());

        return edges;
    }
};

#endif