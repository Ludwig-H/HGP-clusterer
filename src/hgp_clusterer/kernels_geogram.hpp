#pragma once


#include <vector>
#include <utility>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>
#include <random>

#include <Eigen/Dense>


// ==============================================================================
// Fast, Thread-Safe Welzl's Algorithm (using Eigen for Basis solving)
// ==============================================================================
namespace HGP_Numerics {

    // Use stack allocation for small dimensions (up to 32) to avoid heap overhead
    // This is critical because solve_basis is called millions of times.
    using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 32, 1>;
    using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 32, 32>;

    // Solves for the center and squared radius of the sphere defined by boundary points R
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

        size_t n_vecs = m - 1;
        // Use Resize-friendly initialization (won't malloc if size < 32)
        Mat V(dim, n_vecs);
        Vec b(n_vecs);
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

        Mat G = V.transpose() * V;
        Vec y = G.ldlt().solve(0.5 * b);
        Vec local_center = V * y;
        r_sq = local_center.squaredNorm();
        
        for(size_t k=0; k<dim; ++k) {
            center(k) = q0[k] + local_center(k);
        }
    }

    // Recursive Welzl
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

        int p_idx = P[n-1];
        welzl(flat_points, P, R, n-1, dim, center, r_sq);

        double dist_sq = 0.0;
        for(size_t k=0; k<dim; ++k) {
            double d = flat_points[p_idx*dim + k] - center(k);
            dist_sq += d*d;
        }

        if (dist_sq > r_sq + 1e-10) {
            R.push_back(p_idx);
            welzl(flat_points, P, R, n-1, dim, center, r_sq);
            R.pop_back();
        }
    }

    // ==============================================================================
    // Specialized Geometry Kernels (Stack-based, Allocation-free)
    // ==============================================================================

    // Solve MEB for Triangle (3 points) and return Center + RadiusSq
    // Used as fallback for N=4
    inline void get_meb_3(
        const double* p0, const double* p1, const double* p2, 
        size_t dim, 
        Vec& center, double& r_sq
    ) {
        // Relative vectors
        // u = p1 - p0
        // v = p2 - p0
        
        double u_sq = 0, v_sq = 0, dot_uv = 0;
        // w_sq (p2-p1) = u^2 + v^2 - 2u.v
        
        for(size_t k=0; k<dim; ++k) {
            double uk = p1[k] - p0[k];
            double vk = p2[k] - p0[k];
            u_sq += uk * uk;
            v_sq += vk * vk;
            dot_uv += uk * vk;
        }
        
        double w_sq = u_sq + v_sq - 2*dot_uv;

        // Check Obtuse Angles
        // Angle at P0 (between u and v)
        if (dot_uv <= 0) {
            // Center is midpoint of P1-P2 (longest edge opposite to P0)
            // Wait, if angle at P0 is obtuse, P1-P2 is the opposite edge.
            // Center is midpoint of P1, P2.
            // r_sq = |p1-p2|^2 / 4 = w_sq / 4
            for(size_t k=0; k<dim; ++k) center(k) = 0.5 * (p1[k] + p2[k]);
            r_sq = w_sq * 0.25;
            return;
        }
        
        // Angle at P1 (between -u and w=v-u) -> dot = u_sq - dot_uv
        if (u_sq - dot_uv <= 0) {
            // Center is midpoint of P0-P2
            for(size_t k=0; k<dim; ++k) center(k) = 0.5 * (p0[k] + p2[k]);
            r_sq = v_sq * 0.25;
            return;
        }

        // Angle at P2 (between -v and -w=u-v) -> dot = v_sq - dot_uv
        if (v_sq - dot_uv <= 0) {
            // Center is midpoint of P0-P1
            for(size_t k=0; k<dim; ++k) center(k) = 0.5 * (p0[k] + p1[k]);
            r_sq = u_sq * 0.25;
            return;
        }

        // Acute Triangle: Circumcenter
        // O = P0 + a*u + b*v
        // u.(O-P0) = u.u/2  => a*u^2 + b*u.v = u^2/2
        // v.(O-P0) = v.v/2  => a*u.v + b*v^2 = v^2/2
        
        double det = u_sq * v_sq - dot_uv * dot_uv;
        if (det <= 1e-14) {
            // Degenerate (collinear): Midpoint of longest edge (hypotenuse)
            // Since we passed angle checks, this shouldn't strictly happen unless very flat acute
            double max_sq = std::max({u_sq, v_sq, w_sq});
            if (u_sq == max_sq) { // P0-P1
                 for(size_t k=0; k<dim; ++k) center(k) = 0.5 * (p0[k] + p1[k]);
                 r_sq = u_sq * 0.25;
            } else if (v_sq == max_sq) { // P0-P2
                 for(size_t k=0; k<dim; ++k) center(k) = 0.5 * (p0[k] + p2[k]);
                 r_sq = v_sq * 0.25;
            } else { // P1-P2
                 for(size_t k=0; k<dim; ++k) center(k) = 0.5 * (p1[k] + p2[k]);
                 r_sq = w_sq * 0.25;
            }
            return;
        }

        double inv_det = 0.5 / det;
        double rhs_u = 0.5 * u_sq;
        double rhs_v = 0.5 * v_sq;
        
        // Cramer 2x2
        // | u2  uv | |a| = |ru|
        // | uv  v2 | |b| = |rv|
        
        double alpha = (v_sq * u_sq - dot_uv * v_sq); // Wait, rhs are u^2/2 etc.
        // alpha = (rhs_u * v_sq - rhs_v * dot_uv) / det
        alpha = (u_sq * v_sq - v_sq * dot_uv) * inv_det; 
        
        // beta = (u_sq * rhs_v - dot_uv * rhs_u) / det
        double beta = (u_sq * v_sq - u_sq * dot_uv) * inv_det;

        // Recompute standard coords
        for(size_t k=0; k<dim; ++k) {
            center(k) = p0[k] + alpha * (p1[k] - p0[k]) + beta * (p2[k] - p0[k]);
        }
        
        // Radius: Distance from P0 (or center norm)
        // r^2 = |alpha u + beta v|^2
        r_sq = alpha*alpha*u_sq + beta*beta*v_sq + 2*alpha*beta*dot_uv;
    }

    // N=3 wrapper for the interface
    inline double compute_meb_sq_radius_3(const double* flat_points, const int* indices, size_t dim) {
        // Reuse the full solver but discard center
        Vec c(dim);
        double r2;
        const double* p0 = &flat_points[indices[0] * dim];
        const double* p1 = &flat_points[indices[1] * dim];
        const double* p2 = &flat_points[indices[2] * dim];
        get_meb_3(p0, p1, p2, dim, c, r2);
        return r2;
    }

    // Specialized N=4 Solver
    inline double compute_meb_sq_radius_4(const double* flat_points, const int* indices, size_t dim) {
        const double* p0 = &flat_points[indices[0] * dim];
        const double* p1 = &flat_points[indices[1] * dim];
        const double* p2 = &flat_points[indices[2] * dim];
        const double* p3 = &flat_points[indices[3] * dim];

        // 1. Try Circumcenter of Tetrahedron
        // O = P0 + x(p1-p0) + y(p2-p0) + z(p3-p0)
        
        // Gram Matrix entries
        double dot[3][3]; 
        // 0:u, 1:v, 2:w
        const double* vecs[3] = {p1, p2, p3}; // actually p[i]-p0
        
        // Compute dots and norms
        for(int i=0; i<3; ++i) {
            for(int j=i; j<3; ++j) {
                double d = 0;
                for(size_t k=0; k<dim; ++k) {
                    d += (vecs[i][k] - p0[k]) * (vecs[j][k] - p0[k]);
                }
                dot[i][j] = d;
                if(i!=j) dot[j][i] = d;
            }
        }

        Eigen::Matrix3d M;
        Eigen::Vector3d rhs;
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) M(i,j) = dot[i][j];
            rhs(i) = 0.5 * dot[i][i];
        }

        // Solve M * coords = rhs
        // Use LDLT for stability with PSD matrix
        auto solver = M.ldlt();
        if(solver.info() == Eigen::Success && solver.rcond() > 1e-10) {
            Eigen::Vector3d coords = solver.solve(rhs);
            
            // Check containment (Barycentric coords >= 0)
            // lambda_0 = 1 - sum(coords)
            double sum_c = coords.sum();
            if(coords.minCoeff() >= -1e-10 && sum_c <= 1.0 + 1e-10) {
                // Circumcenter is inside!
                // Radius = |x u + y v + z w|^2
                // = coords^T * M * coords
                return coords.transpose() * M * coords;
            }
        }

        // 2. Fallback: Max of Faces
        // If circumcenter is outside (or singular), solution is on boundary.
        // We check all 4 faces.
        // The global MEB is determined by the face MEB that covers the 4th point?
        // No, standard algorithm: 
        // Solution is the MEB of one of the faces.
        // Which one? The one that contains the 4th point. 
        // But simply taking the one with the *largest* radius is a sufficient heuristic 
        // for N=4 in convex position (Tetrahedron) *if* we assume the points are not inside one another.
        // Actually, for a tetrahedron, if the circumcenter is outside, the MEB is unique and lies on a face/edge.
        // Checking all 4 faces and taking the valid one with min radius? No.
        // The MEB is the "smallest enclosing ball".
        // It's the unique ball.
        // Strategy: 
        // Compute MEB of each face F_i.
        // Check if Point P_i (opposite) is inside.
        // If yes, this IS the solution.
        
        // Face 0: P1, P2, P3 (opposite P0)
        {
            Vec c(dim); double r2;
            get_meb_3(p1, p2, p3, dim, c, r2);
            // Check P0 distance
            double d2 = 0; for(size_t k=0; k<dim; ++k) d2 += (p0[k]-c(k))*(p0[k]-c(k));
            if(d2 <= r2 + 1e-10) return r2;
        }
        // Face 1: P0, P2, P3 (opposite P1)
        {
            Vec c(dim); double r2;
            get_meb_3(p0, p2, p3, dim, c, r2);
            double d2 = 0; for(size_t k=0; k<dim; ++k) d2 += (p1[k]-c(k))*(p1[k]-c(k));
            if(d2 <= r2 + 1e-10) return r2;
        }
        // Face 2: P0, P1, P3 (opposite P2)
        {
            Vec c(dim); double r2;
            get_meb_3(p0, p1, p3, dim, c, r2);
            double d2 = 0; for(size_t k=0; k<dim; ++k) d2 += (p2[k]-c(k))*(p2[k]-c(k));
            if(d2 <= r2 + 1e-10) return r2;
        }
        // Face 3: P0, P1, P2 (opposite P3)
        {
            Vec c(dim); double r2;
            get_meb_3(p0, p1, p2, dim, c, r2);
            double d2 = 0; for(size_t k=0; k<dim; ++k) d2 += (p3[k]-c(k))*(p3[k]-c(k));
            if(d2 <= r2 + 1e-10) return r2;
        }

        // If we reach here, it might be numerical noise or degenerate.
        // Return the max radius found? Or fallback to Welzl?
        // Fallback to generic Welzl for safety.
        // But usually one face covers it.
        // Let's fallback to recursive Welzl to be 100% safe against edge cases.
        std::vector<int> P(indices, indices + 4);
        std::vector<int> R; R.reserve(dim+1);
        Vec c(dim); double r_sq=0;
        welzl(flat_points, P, R, 4, dim, c, r_sq);
        return r_sq;
    }

    inline double compute_meb_sq_radius(const double* flat_points, const int* indices, size_t k_size, size_t dim) {
        if (k_size == 0) return 0.0;
        if (k_size == 1) return 0.0;
        if (k_size == 2) {
            double d2 = 0;
            const double* p1 = &flat_points[indices[0]*dim];
            const double* p2 = &flat_points[indices[1]*dim];
            for(size_t k=0; k<dim; ++k) {
                double diff = p1[k] - p2[k];
                d2 += diff*diff;
            }
            return d2 * 0.25;
        }
        if (k_size == 3) {
            return compute_meb_sq_radius_3(flat_points, indices, dim);
        }
        if (k_size == 4) {
            return compute_meb_sq_radius_4(flat_points, indices, dim);
        }

        // Generic Welzl for K > 4
        std::vector<int> P(indices, indices + k_size);
        std::vector<int> R;
        R.reserve(dim + 1);
        
        if (k_size > 10) {
             for (size_t i = P.size() - 1; i > 0; --i) {
                size_t j = (size_t((i * 12345 + 6789)) % (i + 1)); 
                std::swap(P[i], P[j]);
            }
        }

        Vec center(dim);
        double r_sq = 0.0;
        welzl(flat_points, P, R, (int)P.size(), dim, center, r_sq);
        return r_sq;
    }
}


#ifdef HGP_WITH_GEOGRAM

#include <geogram/delaunay/delaunay.h>

inline bool my_cell_is_infinite(const GEO::Delaunay* delaunay, GEO::index_t c) {
    GEO::index_t c_size = delaunay->cell_size();
    for(GEO::index_t i=0; i<c_size; ++i) {
        if(delaunay->cell_vertex(c, i) == GEO::index_t(-1)) return true;
    }
    return false;
}

#include <geogram/basic/common.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/file_system.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_sort.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#endif

#include "kernels.hpp"
// Geogram Implementation
class GeogramDelaunayImpl : public WeightedDelaunayTraits {
public:
    GeogramDelaunayImpl() {
        static bool initialized = false;
        if (!initialized) {
            GEO::initialize();
            GEO::CmdLine::import_arg_group("global");
            GEO::CmdLine::import_arg_group("algo");
            GEO::CmdLine::import_arg_group("standard");
            GEO::CmdLine::set_arg("sys:multithread", "true");
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
        const int* indices,
        size_t k,
        size_t dim
    ) override {
        return HGP_Numerics::compute_meb_sq_radius(flat_points, indices, k, dim);
    }

private:
    mutable std::vector<double> _lifted_buffer;

    std::vector<std::pair<int, int>> _compute_edges(
        const double* flat_points,
        const std::vector<double>* weights_ptr,
        size_t n_points,
        size_t dim
    ) {
        if (n_points < 2) return {};
        
        std::vector<std::pair<int, int>> edges;
        bool weighted = (weights_ptr && !weights_ptr->empty());

        if (!weighted) {
            edges = _compute_standard(flat_points, n_points, dim);
        } else {
            edges = _compute_weighted(flat_points, *weights_ptr, n_points, dim);
        }

        // Deduplicate (Parallel sort could be used here but std::sort is usually fast enough for edges list)
        if (!edges.empty()) {
            #ifdef CGAL_LINKED_WITH_TBB
            tbb::parallel_sort(edges.begin(), edges.end());
            #else
            std::sort(edges.begin(), edges.end());
            #endif
            auto last = std::unique(edges.begin(), edges.end());
            edges.erase(last, edges.end());
        }
        
        return edges;
    }

    std::vector<std::pair<int, int>> _compute_standard(
        const double* flat_points,
        size_t n_points,
        size_t dim
    ) {
        std::vector<std::pair<int, int>> edges;
        
        std::string engine_name = "default";
        const char* env_engine = std::getenv("GEOGRAM_ENGINE_3D");

        if (dim == 2) engine_name = "BDEL2d";
        else if (dim == 3) {
            if (env_engine) engine_name = std::string(env_engine);
            else engine_name = "PDEL"; 
        }
        

        GEO::Delaunay_var delaunay = GEO::Delaunay::create(dim, engine_name);
        if (!delaunay) {
            return edges;
        }

        // Memory Optimization
        delaunay->set_stores_neighbors(false);
        delaunay->set_stores_cicl(false);
        
        // CRITICAL FIX: PDEL (ParallelDelaunay3d) requires reordering to be enabled.
        // If set to false, it expects manual BRIO levels via set_BRIO_levels().
        // Since we don't provide them, disabling reordering causes a SegFault 
        // when PDEL tries to access the uninitialized levels vector.
        delaunay->set_reorder(true);
        
        delaunay->set_vertices(n_points, flat_points);

        GEO::index_t n_cells = delaunay->nb_cells();
        
        if (n_cells > 0) {
            int n_verts_per_cell = delaunay->cell_size();
            _parallel_extract_all_edges(delaunay, n_points, n_cells, n_verts_per_cell, edges);
        }
        
        return edges;
    }

    std::vector<std::pair<int, int>> _compute_weighted(
        const double* flat_points,
        const std::vector<double>& weights,
        size_t n_points,
        size_t dim
    ) {
        std::vector<std::pair<int, int>> edges;
        
        double max_weight = -std::numeric_limits<double>::infinity();
        if (!weights.empty()) {
            for(double w : weights) {
                if(w > max_weight) max_weight = w;
            }
        } else {
            max_weight = 0.0;
        }
        
        size_t lifted_dim = dim + 1;
        
        size_t required_size = n_points * lifted_dim;
        if (_lifted_buffer.size() < required_size) {
            _lifted_buffer.resize(required_size);
        }
        
        double* lifted_ptr = _lifted_buffer.data();
        
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_points), [&](const tbb::blocked_range<size_t>& r) {
            for(size_t i=r.begin(); i!=r.end(); ++i) {
                for(size_t d=0; d<dim; ++d) {
                    lifted_ptr[i*lifted_dim + d] = flat_points[i*dim + d];
                }
                double diff = max_weight - weights[i];
                lifted_ptr[i*lifted_dim + dim] = std::sqrt(diff < 0 ? 0 : diff);
            }
        });
        #else
        for(size_t i=0; i<n_points; ++i) {
            for(size_t d=0; d<dim; ++d) {
                lifted_ptr[i*lifted_dim + d] = flat_points[i*dim + d];
            }
            double diff = max_weight - weights[i];
            lifted_ptr[i*lifted_dim + dim] = std::sqrt(diff < 0 ? 0 : diff);
        }
        #endif

        const char* env_engine = std::getenv("GEOGRAM_ENGINE_3D");
        std::string engine_name = "default";
        
        if (lifted_dim == 3) {
            if (env_engine) engine_name = std::string(env_engine);
            else engine_name = "PDEL";
        } else if (lifted_dim == 4) {
            if (env_engine) engine_name = std::string(env_engine);
            else engine_name = "PDEL"; 
        }
        
        GEO::Delaunay_var delaunay = GEO::Delaunay::create(lifted_dim, engine_name);
        if (!delaunay) {
            return edges;
        }

        // Memory Optimization: Disable auxiliary structures
        delaunay->set_stores_neighbors(false);
        delaunay->set_stores_cicl(false);
        
        // CRITICAL FIX: PDEL (ParallelDelaunay3d) requires reordering to be enabled.
        // If set to false, it expects manual BRIO levels via set_BRIO_levels().
        // Since we don't provide them, disabling reordering causes a SegFault 
        // when PDEL tries to access the uninitialized levels vector.
        delaunay->set_reorder(true);
        
        delaunay->set_vertices(n_points, lifted_ptr);
        
        GEO::index_t n_cells = delaunay->nb_cells();
        
        if (n_cells > 0) {
            int c_size = delaunay->cell_size();
            
            if (c_size == dim + 1) {
                 _parallel_extract_all_edges(delaunay, n_points, n_cells, c_size, edges);
            }
            else if (c_size == dim + 2) {
                if (dim == 2) _extract_lower_hull_2d(delaunay, n_points, n_cells, edges);
                else if (dim == 3) _extract_lower_hull_3d(delaunay, n_points, n_cells, edges);
                else _extract_lower_hull_nd(delaunay, n_points, n_cells, dim, lifted_dim, edges);
            }
        }
        
        // Release memory immediately
        delaunay.reset();
        _lifted_buffer.clear();
        _lifted_buffer.shrink_to_fit();
        
        return edges;
    }

    // ==============================================================================
    // Parallel Extraction Helpers
    // ==============================================================================

    void _parallel_extract_all_edges(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        int c_size,
        std::vector<std::pair<int, int>>& edges
    ) {
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::concurrent_vector<std::pair<int, int>> concurrent_edges;
        
        // Chunk size: Trade-off between local dedup efficiency and load balancing
        size_t chunk_size = 1024; 
        
        tbb::parallel_for(tbb::blocked_range<GEO::index_t>(0, n_cells, chunk_size), 
            [&](const tbb::blocked_range<GEO::index_t>& r) {
            
            std::vector<std::pair<int, int>> local_edges;
            // Heuristic reserve: 6 edges per cell, but we expect dedup to reduce it
            local_edges.reserve((r.end() - r.begin()) * 6);

            for(GEO::index_t c=r.begin(); c!=r.end(); ++c) {
                if(delaunay->keeps_infinite() && my_cell_is_infinite(delaunay.get(), c)) continue;
                
                for(int i=0; i<c_size; ++i) {
                    for(int j=i+1; j<c_size; ++j) {
                        GEO::index_t v1 = delaunay->cell_vertex(c, i);
                        GEO::index_t v2 = delaunay->cell_vertex(c, j);
                        if(v1 < n_points && v2 < n_points) {
                            if(v1 < v2) local_edges.push_back({(int)v1, (int)v2});
                            else local_edges.push_back({(int)v2, (int)v1});
                        }
                    }
                }
            }
            
            // Local Deduplication (Crucial for Memory)
            if(!local_edges.empty()) {
                std::sort(local_edges.begin(), local_edges.end());
                auto last = std::unique(local_edges.begin(), local_edges.end());
                local_edges.erase(last, local_edges.end());
                
                // Flush to global
                auto it = concurrent_edges.grow_by(local_edges.size());
                std::copy(local_edges.begin(), local_edges.end(), it);
            }
        });
        
        // Copy back to std::vector (contiguous)
        edges.assign(concurrent_edges.begin(), concurrent_edges.end());
        
        // Final Global Sort & Unique
        tbb::parallel_sort(edges.begin(), edges.end());
        auto last = std::unique(edges.begin(), edges.end());
        edges.erase(last, edges.end());

        #else
        // Sequential Fallback
        for(GEO::index_t c=0; c<n_cells; ++c) {
            if(delaunay->keeps_infinite() && my_cell_is_infinite(delaunay.get(), c)) continue;
            for(int i=0; i<c_size; ++i) {
                for(int j=i+1; j<c_size; ++j) {
                    GEO::index_t v1 = delaunay->cell_vertex(c, i);
                    GEO::index_t v2 = delaunay->cell_vertex(c, j);
                    if(v1 < n_points && v2 < n_points) {
                        if(v1 < v2) edges.push_back({(int)v1, (int)v2});
                        else edges.push_back({(int)v2, (int)v1});
                    }
                }
            }
        }
        std::sort(edges.begin(), edges.end());
        auto last = std::unique(edges.begin(), edges.end());
        edges.erase(last, edges.end());
        #endif
    }

    void _extract_lower_hull_2d(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        std::vector<std::pair<int, int>>& edges
    ) {
        #ifdef CGAL_LINKED_WITH_TBB
        tbb::concurrent_vector<std::pair<int, int>> concurrent_edges;
        tbb::parallel_for(tbb::blocked_range<GEO::index_t>(0, n_cells), [&](const tbb::blocked_range<GEO::index_t>& r) {
            std::vector<std::pair<int, int>> local_edges;
            
            for(GEO::index_t c=r.begin(); c!=r.end(); ++c) {
        #else
            for(GEO::index_t c = 0; c < n_cells; ++c) {
        #endif
            if (delaunay->keeps_infinite() && my_cell_is_infinite(delaunay.get(), c)) continue;

            for (GEO::index_t f = 0; f < 4; ++f) {
                GEO::index_t adj = delaunay->cell_adjacent(c, f);
                bool is_boundary = (adj == GEO::index_t(-1));
                if (!is_boundary && delaunay->keeps_infinite()) {
                    if (my_cell_is_infinite(delaunay.get(), adj)) is_boundary = true;
                }

                if (is_boundary) {
                    GEO::index_t v_idx[3];
                    int k = 0;
                    for(int j=0; j<4; ++j) {
                        if(j != (int)f) v_idx[k++] = delaunay->cell_vertex(c, j);
                    }
                    
                    const double* p0 = delaunay->vertex_ptr(v_idx[0]);
                    const double* p1 = delaunay->vertex_ptr(v_idx[1]);
                    const double* p2 = delaunay->vertex_ptr(v_idx[2]);
                    
                    double u[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
                    double v[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
                    
                    double nx = u[1]*v[2] - u[2]*v[1];
                    double ny = u[2]*v[0] - u[0]*v[2];
                    double nz = u[0]*v[1] - u[1]*v[0];
                    
                    GEO::index_t v_in = delaunay->cell_vertex(c, f);
                    const double* p_in = delaunay->vertex_ptr(v_in);
                    double dx = p_in[0] - p0[0];
                    double dy = p_in[1] - p0[1];
                    double dz = p_in[2] - p0[2];
                    
                    if (nx*dx + ny*dy + nz*dz > 0) nz = -nz;

                    if (nz < 0) {
                        auto add_edge = [&](GEO::index_t i1, GEO::index_t i2) {
                             if (i1 < n_points && i2 < n_points) {
                                #ifdef CGAL_LINKED_WITH_TBB
                                 if (i1 < i2) local_edges.push_back({(int)i1, (int)i2}); else local_edges.push_back({(int)i2, (int)i1});
                                #else
                                 if (i1 < i2) edges.push_back({(int)i1, (int)i2}); else edges.push_back({(int)i2, (int)i1});
                                #endif
                             }
                        };
                        add_edge(v_idx[0], v_idx[1]);
                        add_edge(v_idx[1], v_idx[2]);
                        add_edge(v_idx[0], v_idx[2]);
                    }
                }
            }
        #ifdef CGAL_LINKED_WITH_TBB
            } // end loop
            if(!local_edges.empty()) {
                auto it = concurrent_edges.grow_by(local_edges.size());
                std::copy(local_edges.begin(), local_edges.end(), it);
            }
        });
        edges.assign(concurrent_edges.begin(), concurrent_edges.end());
        #else
        }
        #endif
    }

    void _extract_lower_hull_3d(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        std::vector<std::pair<int, int>>& edges
    ) {
        // Sequential extraction to save memory (avoid concurrent_vector copy peak)
        for (GEO::index_t c = 0; c < n_cells; ++c) {
            if (delaunay->keeps_infinite() && my_cell_is_infinite(delaunay.get(), c)) continue;

            // Optimization: Precompute sum of weights (lifted coordinate, index 3)
            // Cell has 5 vertices.
            double w_sum = 0.0;
            double v_w[5]; 
            GEO::index_t v_idx[5];
            
            for(int j=0; j<5; ++j) {
                v_idx[j] = delaunay->cell_vertex(c, j);
                // Accessing the 4th coordinate (index 3) of the lifted point
                v_w[j] = delaunay->vertex_ptr(v_idx[j])[3];
                w_sum += v_w[j];
            }

            for (GEO::index_t f = 0; f < 5; ++f) {
                GEO::index_t adj = delaunay->cell_adjacent(c, f);
                bool is_boundary = (adj == GEO::index_t(-1));
                if (!is_boundary && delaunay->keeps_infinite()) {
                    if (my_cell_is_infinite(delaunay.get(), adj)) is_boundary = true;
                }

                if (is_boundary) {
                    // Geometric check: Lower Hull
                    // Centroid of facet (excluding v_f) vs Centroid of cell
                    // H_f < H_c  <=>  Sum(w_others) / 4 < Sum(all) / 5
                    // <=> 5 * (Sum(all) - w_f) < 4 * Sum(all)
                    // <=> Sum(all) < 5 * w_f
                    
                    if (w_sum < 5.0 * v_w[f]) {
                        // Extract edges of the facet (tetrahedron)
                        // Facet vertices are all v_idx[j] where j != f
                        
                        // We have 4 vertices in the facet. 6 Edges.
                        // Vertices indices in `v_idx` array excluding `f`
                        int local_v[4];
                        int k=0;
                        for(int j=0; j<5; ++j) {
                            if(j != f) local_v[k++] = j;
                        }
                        
                        for(int a=0; a<4; ++a) {
                            int idx_a = v_idx[local_v[a]];
                            // Skip invalid points early (though usually checked outside)
                            if (idx_a >= (GEO::index_t)n_points) continue;

                            for(int b=a+1; b<4; ++b) {
                                int idx_b = v_idx[local_v[b]];
                                if (idx_b >= (GEO::index_t)n_points) continue;
                                
                                if (idx_a < idx_b) edges.push_back({(int)idx_a, (int)idx_b});
                                else edges.push_back({(int)idx_b, (int)idx_a});
                            }
                        }
                    }
                }
            }
        }
    }

    void _extract_lower_hull_nd(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        size_t dim,
        size_t lifted_dim,
        std::vector<std::pair<int, int>>& edges
    ) {
        GEO::index_t n_facets_per_cell = delaunay->cell_size();

        #ifdef CGAL_LINKED_WITH_TBB
        tbb::concurrent_vector<std::pair<int, int>> concurrent_edges;
        tbb::parallel_for(tbb::blocked_range<GEO::index_t>(0, n_cells), [&](const tbb::blocked_range<GEO::index_t>& r) {
            std::vector<std::pair<int, int>> local_edges;
            for(GEO::index_t c=r.begin(); c!=r.end(); ++c) {
        #else
        for (GEO::index_t c = 0; c < n_cells; ++c) {
        #endif
            if (delaunay->keeps_infinite() && my_cell_is_infinite(delaunay.get(), c)) continue;

            for (GEO::index_t f = 0; f < n_facets_per_cell; ++f) {
                GEO::index_t adj = delaunay->cell_adjacent(c, f);
                bool is_boundary = (adj == GEO::index_t(-1));
                if (!is_boundary && delaunay->keeps_infinite()) {
                    if (my_cell_is_infinite(delaunay.get(), adj)) is_boundary = true;
                }

                if (is_boundary) {
                    std::vector<GEO::index_t> face_v;
                    face_v.reserve(n_facets_per_cell - 1);
                    for(GEO::index_t i=0; i<n_facets_per_cell; ++i) {
                        if (i != f) face_v.push_back(delaunay->cell_vertex(c, i));
                    }
                    
                    // Simple centroid check (Optimization: avoid std::vector for coords if dim is small)
                    double c_lifted = 0;
                    for(GEO::index_t j=0; j<n_facets_per_cell; ++j) {
                        c_lifted += delaunay->vertex_ptr(delaunay->cell_vertex(c, j))[dim];
                    }
                    c_lifted /= double(n_facets_per_cell);

                    double f_lifted = 0;
                    for(auto v : face_v) {
                        f_lifted += delaunay->vertex_ptr(v)[dim];
                    }
                    f_lifted /= double(face_v.size());

                    if (f_lifted < c_lifted) {
                        size_t f_dim = face_v.size();
                        for(size_t a=0; a<f_dim; ++a) {
                            for(size_t b=a+1; b<f_dim; ++b) {
                                GEO::index_t v1 = face_v[a];
                                GEO::index_t v2 = face_v[b];
                                if(v1 < n_points && v2 < n_points) {
                                    #ifdef CGAL_LINKED_WITH_TBB
                                    if (v1 < v2) local_edges.push_back({(int)v1, (int)v2}); else local_edges.push_back({(int)v2, (int)v1});
                                    #else
                                    if (v1 < v2) edges.push_back({(int)v1, (int)v2}); else edges.push_back({(int)v2, (int)v1});
                                    #endif
                                }
                            }
                        }
                    }
                }
            }
        #ifdef CGAL_LINKED_WITH_TBB
            }
            if(!local_edges.empty()) {
                auto it = concurrent_edges.grow_by(local_edges.size());
                std::copy(local_edges.begin(), local_edges.end(), it);
            }
        });
        edges.assign(concurrent_edges.begin(), concurrent_edges.end());
        #else
        }
        #endif
    }
};

#endif