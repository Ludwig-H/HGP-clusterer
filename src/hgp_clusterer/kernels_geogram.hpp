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

        // =========================================================
        // CASE A: Weighted Delaunay (via Lifting to dim+1)
        // =========================================================
        if (weights_ptr && !weights_ptr->empty()) {
            size_t lifted_dim = dim + 1;
            std::vector<double> lifted_points(n_points * lifted_dim);
            
            // 1. Lift points: (x, ..., ||x||^2 - w)
            for(size_t i=0; i<n_points; ++i) {
                double sq_norm = 0.0;
                for(size_t d=0; d<dim; ++d) {
                    double val = flat_points[i*dim + d];
                    lifted_points[i*lifted_dim + d] = val;
                    sq_norm += val*val;
                }
                lifted_points[i*lifted_dim + dim] = sq_norm - (*weights_ptr)[i];
            }

            // 2. Compute Delaunay in dim+1
            GEO::Delaunay_var delaunay = GEO::Delaunay::create(lifted_dim, "default");
            delaunay->set_vertices(n_points, lifted_points.data());

            GEO::index_t n_cells = delaunay->nb_cells();
            GEO::index_t n_facets_per_cell = delaunay->cell_size(); // = dim + 2

            // 3. Extract Lower Convex Hull Edges
            
            // --- OPTIMIZED PATH FOR 2D INPUT (3D LIFTED) ---
            if (dim == 2) {
                struct Face3 {
                    GEO::index_t v[3]; // Sorted vertices
                    GEO::index_t c;    // Cell index
                    GEO::index_t f;    // Local facet index

                    bool operator<(const Face3& other) const {
                        if (v[0] != other.v[0]) return v[0] < other.v[0];
                        if (v[1] != other.v[1]) return v[1] < other.v[1];
                        return v[2] < other.v[2];
                    }
                    bool operator==(const Face3& other) const {
                        return v[0] == other.v[0] && v[1] == other.v[1] && v[2] == other.v[2];
                    }
                };

                std::vector<Face3> all_faces;
                all_faces.reserve(n_cells * 4); // 4 faces per tet

                for (GEO::index_t c = 0; c < n_cells; ++c) {
                    for (GEO::index_t f = 0; f < 4; ++f) {
                        Face3 face;
                        face.c = c;
                        face.f = f;
                        // Collect vertices excluding f
                        int k = 0;
                        for(int i=0; i<4; ++i) {
                            if(i != (int)f) face.v[k++] = delaunay->cell_vertex(c, i);
                        }
                        // Sort vertices for canonical representation
                        std::sort(std::begin(face.v), std::end(face.v));
                        all_faces.push_back(face);
                    }
                }

                // Sort to group identical faces
                std::sort(all_faces.begin(), all_faces.end());

                // Iterate and find unique faces (boundary)
                for (size_t i = 0; i < all_faces.size(); ++i) {
                    // Check if unique
                    bool is_duplicate = false;
                    if (i + 1 < all_faces.size() && all_faces[i] == all_faces[i+1]) is_duplicate = true;
                    if (i > 0 && all_faces[i] == all_faces[i-1]) is_duplicate = true;

                    if (!is_duplicate) {
                        // Boundary Facet Logic
                        const auto& face = all_faces[i];
                        GEO::index_t c = face.c;
                        GEO::index_t f = face.f;

                        // Re-fetch unsorted vertices for geometry (though sorted works too for normal magnitude, sign matters)
                        // Actually, we need the geometry from the cell to be consistent.
                        
                        // Recover vertices from the cell using local index f
                        GEO::index_t v_idx[3];
                        int k = 0;
                        for(int j=0; j<4; ++j) {
                            if(j != (int)f) v_idx[k++] = delaunay->cell_vertex(c, j);
                        }

                        const double* p0 = &lifted_points[v_idx[0] * 3];
                        const double* p1 = &lifted_points[v_idx[1] * 3];
                        const double* p2 = &lifted_points[v_idx[2] * 3];

                        double u[3] = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
                        double v[3] = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};

                        double nx = u[1]*v[2] - u[2]*v[1];
                        double ny = u[2]*v[0] - u[0]*v[2];
                        double nz = u[0]*v[1] - u[1]*v[0];

                        GEO::index_t v_in = delaunay->cell_vertex(c, f);
                        const double* p_in = &lifted_points[v_in * 3];
                        double dx = p_in[0] - p0[0];
                        double dy = p_in[1] - p0[1];
                        double dz = p_in[2] - p0[2];

                        if (nx*dx + ny*dy + nz*dz > 0) nz = -nz;

                        if (nz < 0) {
                            auto add_edge = [&](GEO::index_t a, GEO::index_t b) {
                                if (a < n_points && b < n_points) {
                                    if (a < b) edges.push_back({(int)a, (int)b});
                                    else edges.push_back({(int)b, (int)a});
                                }
                            };
                            add_edge(v_idx[0], v_idx[1]);
                            add_edge(v_idx[1], v_idx[2]);
                            add_edge(v_idx[0], v_idx[2]);
                        }
                    }
                }
            } 
            // --- OPTIMIZED PATH FOR 3D INPUT (4D LIFTED) ---
            else if (dim == 3) {
                struct Face4 {
                    GEO::index_t v[4];
                    GEO::index_t c;
                    GEO::index_t f;

                    bool operator<(const Face4& other) const {
                        for(int k=0; k<4; ++k) if(v[k] != other.v[k]) return v[k] < other.v[k];
                        return false;
                    }
                    bool operator==(const Face4& other) const {
                        for(int k=0; k<4; ++k) if(v[k] != other.v[k]) return false;
                        return true;
                    }
                };

                std::vector<Face4> all_faces;
                all_faces.reserve(n_cells * 5); // 5 faces per pentatope

                for (GEO::index_t c = 0; c < n_cells; ++c) {
                    for (GEO::index_t f = 0; f < 5; ++f) {
                        Face4 face;
                        face.c = c;
                        face.f = f;
                        int k = 0;
                        for(int i=0; i<5; ++i) {
                            if(i != (int)f) face.v[k++] = delaunay->cell_vertex(c, i);
                        }
                        std::sort(std::begin(face.v), std::end(face.v));
                        all_faces.push_back(face);
                    }
                }

                std::sort(all_faces.begin(), all_faces.end());

                for (size_t i = 0; i < all_faces.size(); ++i) {
                    bool is_duplicate = false;
                    if (i + 1 < all_faces.size() && all_faces[i] == all_faces[i+1]) is_duplicate = true;
                    if (i > 0 && all_faces[i] == all_faces[i-1]) is_duplicate = true;

                    if (!is_duplicate) {
                        const auto& face = all_faces[i];
                        GEO::index_t c = face.c;
                        // GEO::index_t f = face.f; // unused variable

                        // Recalculate centroids
                        double cell_c[4] = {0, 0, 0, 0};
                        GEO::index_t c_verts[5];
                        for(int j=0; j<5; ++j) {
                            GEO::index_t v = delaunay->cell_vertex(c, j);
                            c_verts[j] = v;
                            const double* p = &lifted_points[v * 4];
                            for(int d=0; d<4; ++d) cell_c[d] += p[d];
                        }
                        for(int d=0; d<4; ++d) cell_c[d] /= 5.0;

                        double facet_c[4] = {0, 0, 0, 0};
                        // Use the sorted vertices from face.v is fine for centroid
                        for(int j=0; j<4; ++j) {
                            const double* p = &lifted_points[face.v[j] * 4];
                            for(int d=0; d<4; ++d) facet_c[d] += p[d];
                        }
                        for(int d=0; d<4; ++d) facet_c[d] *= 0.25;

                        if (facet_c[3] < cell_c[3]) {
                            auto add_edge = [&](GEO::index_t a, GEO::index_t b) {
                                if (a < n_points && b < n_points) {
                                    if (a < b) edges.push_back({(int)a, (int)b});
                                    else edges.push_back({(int)b, (int)a});
                                }
                            };
                            add_edge(face.v[0], face.v[1]);
                            add_edge(face.v[0], face.v[2]);
                            add_edge(face.v[0], face.v[3]);
                            add_edge(face.v[1], face.v[2]);
                            add_edge(face.v[1], face.v[3]);
                            add_edge(face.v[2], face.v[3]);
                        }
                    }
                }
            }
            // --- GENERIC PATH FOR N-D INPUT ---
            else {
                struct FaceND {
                    std::vector<GEO::index_t> v;
                    GEO::index_t c;
                    GEO::index_t f;

                    bool operator<(const FaceND& other) const {
                        if (v.size() != other.v.size()) return v.size() < other.v.size();
                        return v < other.v;
                    }
                    bool operator==(const FaceND& other) const {
                        return v == other.v;
                    }
                };

                std::vector<FaceND> all_faces;
                all_faces.reserve(n_cells * (dim + 2)); 

                for (GEO::index_t c = 0; c < n_cells; ++c) {
                    for (GEO::index_t f = 0; f < n_facets_per_cell; ++f) { // facets = dim+2
                        FaceND face;
                        face.c = c;
                        face.f = f;
                        face.v.reserve(n_facets_per_cell - 1);
                        
                        for(GEO::index_t i = 0; i < n_facets_per_cell; ++i) {
                            if (i != f) face.v.push_back(delaunay->cell_vertex(c, i));
                        }
                        std::sort(face.v.begin(), face.v.end());
                        all_faces.push_back(face);
                    }
                }

                std::sort(all_faces.begin(), all_faces.end());

                for (size_t i = 0; i < all_faces.size(); ++i) {
                    bool is_duplicate = false;
                    if (i + 1 < all_faces.size() && all_faces[i] == all_faces[i+1]) is_duplicate = true;
                    if (i > 0 && all_faces[i] == all_faces[i-1]) is_duplicate = true;

                    if (!is_duplicate) {
                        const auto& face = all_faces[i];
                        GEO::index_t c = face.c;
                        
                        // Centroids
                        std::vector<double> cell_centroid(lifted_dim, 0.0);
                        for(GEO::index_t j=0; j<n_facets_per_cell; ++j) {
                            GEO::index_t v_idx = delaunay->cell_vertex(c, j);
                            const double* p = &lifted_points[v_idx * lifted_dim];
                            for(size_t d=0; d<lifted_dim; ++d) cell_centroid[d] += p[d];
                        }
                        double inv_c = 1.0 / double(n_facets_per_cell);
                        for(size_t d=0; d<lifted_dim; ++d) cell_centroid[d] *= inv_c;

                        std::vector<double> facet_centroid(lifted_dim, 0.0);
                        for(auto v_idx : face.v) {
                            const double* p = &lifted_points[v_idx * lifted_dim];
                            for(size_t d=0; d<lifted_dim; ++d) facet_centroid[d] += p[d];
                        }
                        double inv_f = 1.0 / double(face.v.size());
                        for(size_t d=0; d<lifted_dim; ++d) facet_centroid[d] *= inv_f;

                        // Lower Hull Test
                        if (facet_centroid[dim] < cell_centroid[dim]) {
                            // Add all edges of this simplex facet
                            size_t f_dim = face.v.size();
                            for(size_t a=0; a<f_dim; ++a) {
                                for(size_t b=a+1; b<f_dim; ++b) {
                                    GEO::index_t v1 = face.v[a];
                                    GEO::index_t v2 = face.v[b];
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

            // Deduplicate
            std::sort(edges.begin(), edges.end());
            auto last = std::unique(edges.begin(), edges.end());
            edges.erase(last, edges.end());
            
            return edges;
        }

        // =========================================================
        // CASE B: Standard Delaunay (No weights)
        // =========================================================
        
        // 1. Create Delaunay
        // Use "default" to let Geogram pick best impl (e.g. parallel BPOW3 in 3D)
        GEO::Delaunay_var delaunay = GEO::Delaunay::create(dim, "default");
        
        // 2. Setup vertices (triggers computation usually)
        delaunay->set_vertices(n_points, flat_points);

        // 3. Extract Edges (Parallel-friendly traversal structure)
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