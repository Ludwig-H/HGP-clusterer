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
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/file_system.h>

#ifdef CGAL_LINKED_WITH_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#endif

// ==============================================================================
// Fast, Thread-Safe Welzl's Algorithm (using Eigen for Basis solving)
// ==============================================================================
namespace HGP_Numerics {

    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

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

    // Optimized solver for Triangle (3 points in dim dimensions)
    inline double compute_meb_sq_radius_3(const double* flat_points, const int* indices, size_t dim) {
        const double* p0 = &flat_points[indices[0] * dim];
        const double* p1 = &flat_points[indices[1] * dim];
        const double* p2 = &flat_points[indices[2] * dim];

        // Edges vectors
        // u = p1 - p0
        // v = p2 - p0
        // w = p2 - p1
        
        double u_sq = 0, v_sq = 0, w_sq = 0;
        double dot_uv = 0;

        for(size_t k=0; k<dim; ++k) {
            double u_k = p1[k] - p0[k];
            double v_k = p2[k] - p0[k];
            double w_k = p2[k] - p1[k];
            u_sq += u_k * u_k;
            v_sq += v_k * v_k;
            w_sq += w_k * w_k;
            dot_uv += u_k * v_k;
        }

        // Check for obtuse angles (dot products < 0 means angle > 90)
        // Angle at P0: dot(p1-p0, p2-p0) = dot_uv
        if (dot_uv <= 0) return std::max(u_sq, v_sq) * 0.25; // Obtuse at P0 -> Longest edge from P0

        // Angle at P1: dot(p0-p1, p2-p1) = dot(-u, w) = -u.w = -u.(v-u) = u.u - u.v
        if (u_sq - dot_uv <= 0) return std::max(u_sq, w_sq) * 0.25; // Obtuse at P1

        // Angle at P2: dot(p0-p2, p1-p2) = dot(-v, -w) = v.w = v.(v-u) = v.v - v.u
        if (v_sq - dot_uv <= 0) return std::max(v_sq, w_sq) * 0.25; // Obtuse at P2

        // Acute triangle: Circumradius
        // R = abc / 4K where K is area
        // a=sqrt(w_sq), b=sqrt(v_sq), c=sqrt(u_sq)
        // 16 K^2 = 4 u^2 v^2 - (u^2 + v^2 - w^2)^2  (Heron-like)
        // Actually simpler with sin formula: R = a / 2sinA
        // Or vector formula: R = ||u|| ||v|| ||u-v|| / 2 ||u x v|| (in 3D)
        // General dim:
        // Denom = 4 * (u^2 v^2 - (u.v)^2)
        double denom = 4.0 * (u_sq * v_sq - dot_uv * dot_uv);
        if (denom <= 1e-14) return std::max({u_sq, v_sq, w_sq}) * 0.25; // Collinear fallback

        double num = u_sq * v_sq * w_sq;
        return num / denom;
    }

    // Stack-based Welzl for small N (up to 4) to avoid heap allocation
    // Simplified version
    inline double compute_meb_sq_radius_small(const double* flat_points, const int* indices, size_t k_size, size_t dim) {
        // Fallback to Eigen-based Welzl but using fixed-size stack arrays
        // Actually, generic welzl creates vectors.
        // For N=4, we just use the existing generic Welzl but we can optimize the vector usage?
        // Or simply trust the generic one is "fast enough" if we avoid allocation.
        // But solve_basis uses Eigen MatrixXd which allocates.
        // Let's stick to the generic one for N >= 4 for now, but N=3 optimization is huge for K=2.
        
        // TODO: A truly allocation-free Welzl for N=4 would be better.
        // For now, redirecting N=3 is the biggest win.
        return 0.0; 
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

        // Generic Welzl for K >= 4
        std::vector<int> P(indices, indices + k_size);
        std::vector<int> R;
        R.reserve(dim + 1);
        
        // Randomized heuristics for stability
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

        // Deduplicate
        if (!edges.empty()) {
            std::sort(edges.begin(), edges.end());
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
        if (dim == 2) engine_name = "BDEL2d";
        else if (dim == 3) engine_name = "BDEL";

        GEO::Delaunay_var delaunay = GEO::Delaunay::create(dim, engine_name);
        if (!delaunay) return edges;

        delaunay->set_reorder(false);
        delaunay->set_vertices(n_points, flat_points);

        GEO::index_t n_cells = delaunay->nb_cells();
        
        // Strategy 1: If cells are available (2D/3D), use them
        if (n_cells > 0) {
            int n_verts_per_cell = delaunay->cell_size();
            for(GEO::index_t c = 0; c < n_cells; ++c) {
                if (delaunay->keeps_infinite() && delaunay->cell_is_infinite(c)) continue;

                for(int i=0; i<n_verts_per_cell; ++i) {
                    for(int j=i+1; j<n_verts_per_cell; ++j) {
                        GEO::index_t v1 = delaunay->cell_vertex(c, i);
                        GEO::index_t v2 = delaunay->cell_vertex(c, j);
                        
                        if (v1 < n_points && v2 < n_points) {
                            if (v1 < v2) edges.push_back({(int)v1, (int)v2});
                            else edges.push_back({(int)v2, (int)v1});
                        }
                    }
                }
            }
        } 
        // Strategy 2: If no cells (nD fallback to NN), use neighbors graph
        else {
            GEO::vector<GEO::index_t> neighbors;
            for(GEO::index_t i = 0; i < n_points; ++i) {
                delaunay->get_neighbors(i, neighbors);
                for(GEO::index_t n_idx : neighbors) {
                    if (n_idx < n_points && n_idx != i) {
                        if (i < n_idx) edges.push_back({(int)i, (int)n_idx});
                        else edges.push_back({(int)n_idx, (int)i});
                    }
                }
            }
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
        
        // Lift points according to Geogram's expected input for BPOW:
        // coords[dim] = sqrt(W - w_i)
        
        double max_weight = -std::numeric_limits<double>::infinity();
        if (!weights.empty()) {
            for(double w : weights) {
                if(w > max_weight) max_weight = w;
            }
        } else {
            max_weight = 0.0;
        }
        
        size_t lifted_dim = dim + 1;
        
        // Re-use buffer if possible
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

        std::string engine_name = "default";
        if (dim == 2) engine_name = "BPOW2d";
        else if (dim == 3) engine_name = "BPOW";
        
        GEO::Delaunay_var delaunay = GEO::Delaunay::create(lifted_dim, engine_name);
        if (!delaunay) {
            std::cerr << "[Geogram Error] Failed to create engine: " << engine_name << std::endl;
            return edges;
        }

        delaunay->set_reorder(false);
        delaunay->set_vertices(n_points, lifted_ptr);
        
        GEO::index_t n_cells = delaunay->nb_cells();
        
        if (n_cells > 0) {
            int c_size = delaunay->cell_size();
            
            // Case 1: The engine returned the Triangulation directly (dim == cell_size - 1)
            if (c_size == dim + 1) {
                 _extract_all_edges(delaunay, n_points, n_cells, c_size, edges);
            }
            // Case 2: The engine returned a Lifted Triangulation (Convex Hull in dim+1)
            else if (c_size == dim + 2) {
                if (dim == 2) _extract_lower_hull_2d(delaunay, n_points, n_cells, edges);
                else if (dim == 3) _extract_lower_hull_3d(delaunay, n_points, n_cells, edges);
                else _extract_lower_hull_nd(delaunay, n_points, n_cells, dim, lifted_dim, edges);
            }
        }
        return edges;
    }

    void _extract_all_edges(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        int c_size,
        std::vector<std::pair<int, int>>& edges
    ) {
        for(GEO::index_t c=0; c<n_cells; ++c) {
            if(delaunay->keeps_infinite() && delaunay->cell_is_infinite(c)) continue;
            
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
    }

    void _extract_lower_hull_2d(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        std::vector<std::pair<int, int>>& edges
    ) {
        // Iterate over all cells (tetrahedra) in the 3D lifted triangulation
        for (GEO::index_t c = 0; c < n_cells; ++c) {
            // Check if cell is infinite (should not happen if we iterate only finite cells, 
            // but standard Geogram nb_cells() might include them depending on impl)
            if (delaunay->keeps_infinite() && delaunay->cell_is_infinite(c)) continue;

            // 3D cell has 4 facets
            for (GEO::index_t f = 0; f < 4; ++f) {
                // Check adjacency
                GEO::index_t adj = delaunay->cell_adjacent(c, f);
                
                // If adjacent is NO_INDEX (-1) or Infinite, then this facet is on the boundary of the convex hull
                bool is_boundary = (adj == GEO::index_t(-1));
                if (!is_boundary && delaunay->keeps_infinite()) {
                    if (delaunay->cell_is_infinite(adj)) is_boundary = true;
                }

                if (is_boundary) {
                    // Extract vertices of the facet
                    GEO::index_t v_idx[3];
                    int k = 0;
                    for(int j=0; j<4; ++j) {
                        if(j != (int)f) v_idx[k++] = delaunay->cell_vertex(c, j);
                    }
                    
                    // Geometric check: Is it facing down?
                    const double* p0 = delaunay->vertex_ptr(v_idx[0]);
                    const double* p1 = delaunay->vertex_ptr(v_idx[1]);
                    const double* p2 = delaunay->vertex_ptr(v_idx[2]);
                    
                    double u[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
                    double v[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
                    
                    double nx = u[1]*v[2] - u[2]*v[1];
                    double ny = u[2]*v[0] - u[0]*v[2];
                    double nz = u[0]*v[1] - u[1]*v[0];
                    
                    // Orient normal outwards
                    GEO::index_t v_in = delaunay->cell_vertex(c, f); // The vertex opposite to the facet
                    const double* p_in = delaunay->vertex_ptr(v_in);
                    double dx = p_in[0] - p0[0];
                    double dy = p_in[1] - p0[1];
                    double dz = p_in[2] - p0[2];
                    
                    if (nx*dx + ny*dy + nz*dz > 0) {
                        // If dot product > 0, normal points inwards (towards v_in)
                        // We want outward normal.
                        nz = -nz;
                    }

                    // Lower hull check: Normal z component is negative
                    if (nz < 0) {
                        // Add edges of this triangle (projected to 2D -> just indices)
                        // We must ensure indices are valid input points (finite)
                        if (v_idx[0] < n_points && v_idx[1] < n_points) {
                            if (v_idx[0] < v_idx[1]) edges.push_back({(int)v_idx[0], (int)v_idx[1]}); else edges.push_back({(int)v_idx[1], (int)v_idx[0]});
                        }
                        if (v_idx[1] < n_points && v_idx[2] < n_points) {
                            if (v_idx[1] < v_idx[2]) edges.push_back({(int)v_idx[1], (int)v_idx[2]}); else edges.push_back({(int)v_idx[2], (int)v_idx[1]});
                        }
                        if (v_idx[0] < n_points && v_idx[2] < n_points) {
                            if (v_idx[0] < v_idx[2]) edges.push_back({(int)v_idx[0], (int)v_idx[2]}); else edges.push_back({(int)v_idx[2], (int)v_idx[0]});
                        }
                    }
                }
            }
        }
    }

    void _extract_lower_hull_3d(
        GEO::Delaunay_var& delaunay,
        size_t n_points,
        GEO::index_t n_cells,
        std::vector<std::pair<int, int>>& edges
    ) {
        for (GEO::index_t c = 0; c < n_cells; ++c) {
            if (delaunay->keeps_infinite() && delaunay->cell_is_infinite(c)) continue;

            // 4D cell (pentatope) has 5 facets (tetrahedra)
            for (GEO::index_t f = 0; f < 5; ++f) {
                GEO::index_t adj = delaunay->cell_adjacent(c, f);
                bool is_boundary = (adj == GEO::index_t(-1));
                if (!is_boundary && delaunay->keeps_infinite()) {
                    if (delaunay->cell_is_infinite(adj)) is_boundary = true;
                }

                if (is_boundary) {
                    // Gather vertices of the facet
                    std::vector<GEO::index_t> face_v; 
                    face_v.reserve(4);
                    for(int j=0; j<5; ++j) {
                        if(j != (int)f) face_v.push_back(delaunay->cell_vertex(c, j));
                    }

                    // Geometric check: Centroid comparison
                    // (Simpler than computing 4D normal)
                    
                    double cell_c[4] = {0, 0, 0, 0};
                    for(int j=0; j<5; ++j) {
                        GEO::index_t v = delaunay->cell_vertex(c, j);
                        const double* p = delaunay->vertex_ptr(v);
                        for(int d=0; d<4; ++d) cell_c[d] += p[d];
                    }
                    for(int d=0; d<4; ++d) cell_c[d] /= 5.0;

                    double facet_c[4] = {0, 0, 0, 0};
                    for(auto v : face_v) {
                        const double* p = delaunay->vertex_ptr(v);
                        for(int d=0; d<4; ++d) facet_c[d] += p[d];
                    }
                    for(int d=0; d<4; ++d) facet_c[d] *= 0.25;

                    // Lower hull check: Facet centroid is "below" cell centroid in lifted dim (index 3)
                    if (facet_c[3] < cell_c[3]) {
                        // Edges of the tetrahedron (facet)
                        size_t f_dim = 4; // 4 vertices
                        for(size_t a=0; a<f_dim; ++a) {
                            for(size_t b=a+1; b<f_dim; ++b) {
                                GEO::index_t v1 = face_v[a];
                                GEO::index_t v2 = face_v[b];
                                if(v1 < n_points && v2 < n_points) {
                                    if (v1 < v2) edges.push_back({(int)v1, (int)v2});
                                    else edges.push_back({(int)v2, (int)v1});
                                }
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
        GEO::index_t n_facets_per_cell = delaunay->cell_size(); // should be lifted_dim + 1

        for (GEO::index_t c = 0; c < n_cells; ++c) {
            if (delaunay->keeps_infinite() && delaunay->cell_is_infinite(c)) continue;

            for (GEO::index_t f = 0; f < n_facets_per_cell; ++f) {
                GEO::index_t adj = delaunay->cell_adjacent(c, f);
                bool is_boundary = (adj == GEO::index_t(-1));
                if (!is_boundary && delaunay->keeps_infinite()) {
                    if (delaunay->cell_is_infinite(adj)) is_boundary = true;
                }

                if (is_boundary) {
                    // Gather vertices
                    std::vector<GEO::index_t> face_v;
                    face_v.reserve(n_facets_per_cell - 1);
                    for(GEO::index_t i=0; i<n_facets_per_cell; ++i) {
                        if (i != f) face_v.push_back(delaunay->cell_vertex(c, i));
                    }
                    
                    // Geometric check: Centroid comparison
                    std::vector<double> cell_centroid(lifted_dim, 0.0);
                    for(GEO::index_t j=0; j<n_facets_per_cell; ++j) {
                        GEO::index_t v_idx = delaunay->cell_vertex(c, j);
                        const double* p = delaunay->vertex_ptr(v_idx);
                        for(size_t d=0; d<lifted_dim; ++d) cell_centroid[d] += p[d];
                    }
                    double inv_c = 1.0 / double(n_facets_per_cell);
                    for(size_t d=0; d<lifted_dim; ++d) cell_centroid[d] *= inv_c;

                    std::vector<double> facet_centroid(lifted_dim, 0.0);
                    for(auto v_idx : face_v) {
                        const double* p = delaunay->vertex_ptr(v_idx);
                        for(size_t d=0; d<lifted_dim; ++d) facet_centroid[d] += p[d];
                    }
                    double inv_f = 1.0 / double(face_v.size());
                    for(size_t d=0; d<lifted_dim; ++d) facet_centroid[d] *= inv_f;

                    // Lower hull check
                    if (facet_centroid[dim] < cell_centroid[dim]) {
                        size_t f_dim = face_v.size();
                        for(size_t a=0; a<f_dim; ++a) {
                            for(size_t b=a+1; b<f_dim; ++b) {
                                GEO::index_t v1 = face_v[a];
                                GEO::index_t v2 = face_v[b];
                                if(v1 < n_points && v2 < n_points) {
                                    if (v1 < v2) edges.push_back({(int)v1, (int)v2});
                                    else edges.push_back({(int)v2, (int)v1});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

#endif