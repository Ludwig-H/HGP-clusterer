#pragma once

#include <vector>
#include <utility>
#include <iostream>
#include <map>
#include <memory>

// Interface for weighted Delaunay edge extraction
// This interface is agnostic of the backend (CGAL, Geogram, etc.)
struct WeightedDelaunayTraits {
    // 1. Weighted (Regular) Triangulation for k > 1
    virtual std::vector<std::pair<int, int>> get_finite_edges(
        const double* flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t dim
    ) = 0;

    // Helper for std::vector compatibility
    std::vector<std::pair<int, int>> get_finite_edges(
        const std::vector<double>& flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t dim
    ) {
        return get_finite_edges(flat_points.data(), weights, n_points, dim);
    }

    // 2. Standard (Unweighted) Delaunay for k = 1 (Optimization)
    virtual std::vector<std::pair<int, int>> get_standard_delaunay_edges(
        const double* flat_points,
        size_t n_points,
        size_t dim
    ) = 0;

    // 3. Radius computation (Welzl's algorithm)
    virtual double compute_simplex_squared_radius(
        const double* flat_points,
        const int* indices,
        size_t k,
        size_t dim
    ) = 0;

    virtual ~WeightedDelaunayTraits() {}
};

#ifdef HGP_WITH_CGAL

// CGAL Headers
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
// Regular (Weighted)
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_vertex_base_2.h>
#include <CGAL/Regular_triangulation_face_base_2.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>

// Standard (Unweighted)
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_3.h>
#include <CGAL/Triangulation_cell_base_3.h>

// Info traits
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

// dD Headers (Partial support)
#include <CGAL/Epick_d.h>
#include <CGAL/Epeck_d.h>
#include <CGAL/Delaunay_triangulation.h>
#include <CGAL/Regular_triangulation.h>
#include <CGAL/Triangulation_vertex.h>
#include <CGAL/Triangulation_data_structure.h>
#include <CGAL/Min_sphere_d.h>
#include <CGAL/Min_sphere_annulus_d_traits_d.h>
#include <CGAL/Min_sphere_annulus_d_traits_2.h>
#include <CGAL/Min_sphere_annulus_d_traits_3.h>

// TBB
#ifdef CGAL_LINKED_WITH_TBB
#include <CGAL/Triangulation_data_structure_3.h>
#endif

// 2D Implementation
template <typename Kernel>
struct WeightedDelaunay2D : public WeightedDelaunayTraits {
    // --- Regular Types ---
    using Vb0_R = CGAL::Regular_triangulation_vertex_base_2<Kernel>;
    using Vb_R  = CGAL::Triangulation_vertex_base_with_info_2<int, Kernel, Vb0_R>;
    using Fb_R  = CGAL::Regular_triangulation_face_base_2<Kernel>;
    using Tds_R = CGAL::Triangulation_data_structure_2<Vb_R, Fb_R>;
    using Rt    = CGAL::Regular_triangulation_2<Kernel, Tds_R>;
    using Weighted_point = typename Kernel::Weighted_point_2;
    using Point_2 = typename Kernel::Point_2;

    // --- Min Sphere Types ---
    using Min_sphere_traits = CGAL::Min_sphere_annulus_d_traits_2<Kernel>;
    using Min_sphere = CGAL::Min_sphere_d<Min_sphere_traits>;

    // --- Standard Types ---
    using Vb0_S = CGAL::Triangulation_vertex_base_2<Kernel>;
    using Vb_S  = CGAL::Triangulation_vertex_base_with_info_2<int, Kernel, Vb0_S>;
    using Fb_S  = CGAL::Triangulation_face_base_2<Kernel>;
    using Tds_S = CGAL::Triangulation_data_structure_2<Vb_S, Fb_S>;
    using Dt    = CGAL::Delaunay_triangulation_2<Kernel, Tds_S>;

    // Implementation for Regular
    std::vector<std::pair<int, int>> get_finite_edges(
        const double* flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t /*dim*/
    ) override {
        std::vector<std::pair<Weighted_point, int>> inputs;
        inputs.reserve(n_points);
        for(size_t i=0; i<n_points; ++i) {
            inputs.emplace_back(
                Weighted_point(Point_2(flat_points[2*i], flat_points[2*i+1]), weights[i]), 
                (int)i
            );
        }
        Rt rt;
        rt.insert(inputs.begin(), inputs.end());
        std::vector<std::pair<int, int>> edges;
        for(auto eit = rt.finite_edges_begin(); eit != rt.finite_edges_end(); ++eit) {
            auto face = eit->first;
            int idx = eit->second;
            auto v1 = face->vertex(Rt::cw(idx));
            auto v2 = face->vertex(Rt::ccw(idx));
            if(!rt.is_infinite(v1) && !rt.is_infinite(v2)) {
                edges.push_back({v1->info(), v2->info()});
            }
        }
        return edges;
    }

    // Implementation for Standard
    std::vector<std::pair<int, int>> get_standard_delaunay_edges(
        const double* flat_points,
        size_t n_points,
        size_t /*dim*/
    ) override {
        std::vector<std::pair<Point_2, int>> inputs;
        inputs.reserve(n_points);
        for(size_t i=0; i<n_points; ++i) {
            inputs.emplace_back(
                Point_2(flat_points[2*i], flat_points[2*i+1]), 
                (int)i
            );
        }
        Dt dt;
        dt.insert(inputs.begin(), inputs.end());
        std::vector<std::pair<int, int>> edges;
        for(auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
            auto face = eit->first;
            int idx = eit->second;
            auto v1 = face->vertex(Dt::cw(idx));
            auto v2 = face->vertex(Dt::ccw(idx));
            if(!dt.is_infinite(v1) && !dt.is_infinite(v2)) {
                edges.push_back({v1->info(), v2->info()});
            }
        }
        return edges;
    }

    double compute_simplex_squared_radius(
        const double* flat_points,
        const int* indices,
        size_t k,
        size_t /*dim*/
    ) override {
        std::vector<Point_2> pts;
        pts.reserve(k);
        for(size_t i=0; i<k; ++i) {
            int idx = indices[i];
            pts.emplace_back(flat_points[2*idx], flat_points[2*idx+1]);
        }
        Min_sphere ms(pts.begin(), pts.end());
        return CGAL::to_double(ms.squared_radius());
    }
};

// 3D Implementation
template <typename Kernel>
struct WeightedDelaunay3D : public WeightedDelaunayTraits {
    using Point_3 = typename Kernel::Point_3;
    using Weighted_point = typename Kernel::Weighted_point_3;

    // --- Regular Types ---
    using Vb0_R = CGAL::Regular_triangulation_vertex_base_3<Kernel>;
    using Vb_R  = CGAL::Triangulation_vertex_base_with_info_3<int, Kernel, Vb0_R>;
    using Cb_R  = CGAL::Regular_triangulation_cell_base_3<Kernel>;
    #ifdef CGAL_LINKED_WITH_TBB
    using Tds_R = CGAL::Triangulation_data_structure_3<Vb_R, Cb_R, CGAL::Parallel_tag>;
    #else
    using Tds_R = CGAL::Triangulation_data_structure_3<Vb_R, Cb_R>;
    #endif
    using Rt    = CGAL::Regular_triangulation_3<Kernel, Tds_R>;

    // --- Min Sphere Types ---
    using Min_sphere_traits = CGAL::Min_sphere_annulus_d_traits_3<Kernel>;
    using Min_sphere = CGAL::Min_sphere_d<Min_sphere_traits>;

    // --- Standard Types ---
    using Vb0_S = CGAL::Triangulation_vertex_base_3<Kernel>;
    using Vb_S  = CGAL::Triangulation_vertex_base_with_info_3<int, Kernel, Vb0_S>;
    using Cb_S  = CGAL::Triangulation_cell_base_3<Kernel>;
    #ifdef CGAL_LINKED_WITH_TBB
    using Tds_S = CGAL::Triangulation_data_structure_3<Vb_S, Cb_S, CGAL::Parallel_tag>;
    #else
    using Tds_S = CGAL::Triangulation_data_structure_3<Vb_S, Cb_S>;
    #endif
    using Dt    = CGAL::Delaunay_triangulation_3<Kernel, Tds_S>;


    std::vector<std::pair<int, int>> get_finite_edges(
        const double* flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t /*dim*/
    ) override {
        std::vector<std::pair<Weighted_point, int>> inputs;
        inputs.reserve(n_points);
        for(size_t i=0; i<n_points; ++i) {
            inputs.emplace_back(
                Weighted_point(Point_3(flat_points[3*i], flat_points[3*i+1], flat_points[3*i+2]), weights[i]), 
                (int)i
            );
        }
        #ifdef CGAL_LINKED_WITH_TBB
        Rt rt(typename Rt::Geom_traits(), nullptr); 
        #else
        Rt rt;
        #endif
        rt.insert(inputs.begin(), inputs.end());
        std::vector<std::pair<int, int>> edges;
        for(auto eit = rt.finite_edges_begin(); eit != rt.finite_edges_end(); ++eit) {
            auto cell = eit->first;
            int i = eit->second;
            int j = eit->third;
            auto v1 = cell->vertex(i);
            auto v2 = cell->vertex(j);
            if (!rt.is_infinite(v1) && !rt.is_infinite(v2)) {
                edges.push_back({v1->info(), v2->info()});
            }
        }
        return edges;
    }

    std::vector<std::pair<int, int>> get_standard_delaunay_edges(
        const double* flat_points, 
        size_t n_points,
        size_t /*dim*/
    ) override {
        std::vector<std::pair<Point_3, int>> inputs;
        inputs.reserve(n_points);
        for(size_t i=0; i<n_points; ++i) {
            inputs.emplace_back(
                Point_3(flat_points[3*i], flat_points[3*i+1], flat_points[3*i+2]), 
                (int)i
            );
        }
        #ifdef CGAL_LINKED_WITH_TBB
        Dt dt(typename Dt::Geom_traits(), nullptr); 
        #else
        Dt dt;
        #endif
        dt.insert(inputs.begin(), inputs.end());
        std::vector<std::pair<int, int>> edges;
        for(auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
            auto cell = eit->first;
            int i = eit->second;
            int j = eit->third;
            auto v1 = cell->vertex(i);
            auto v2 = cell->vertex(j);
            if (!dt.is_infinite(v1) && !dt.is_infinite(v2)) {
                edges.push_back({v1->info(), v2->info()});
            }
        }
        return edges;
    }

    double compute_simplex_squared_radius(
        const double* flat_points,
        const int* indices,
        size_t k,
        size_t /*dim*/
    ) override {
        std::vector<Point_3> pts;
        pts.reserve(k);
        for(size_t i=0; i<k; ++i) {
            int idx = indices[i];
            pts.emplace_back(flat_points[3*idx], flat_points[3*idx+1], flat_points[3*idx+2]);
        }
        Min_sphere ms(pts.begin(), pts.end());
        return CGAL::to_double(ms.squared_radius());
    }
};

// dD Implementation
template <typename Kernel>
struct WeightedDelaunayDD : public WeightedDelaunayTraits {
    using Point_d = typename Kernel::Point_d;
    using Weighted_point_d = typename Kernel::Weighted_point_d;
    
    // Use default types provided by CGAL for dD to avoid TDS configuration errors
    using Rt = CGAL::Regular_triangulation<Kernel>;
    using Dt = CGAL::Delaunay_triangulation<Kernel>;
    
    // Min Sphere Types for dD
    using Min_sphere_traits = CGAL::Min_sphere_annulus_d_traits_d<Kernel>;
    using Min_sphere = CGAL::Min_sphere_d<Min_sphere_traits>;

    std::vector<std::pair<int, int>> get_finite_edges(
        const double* flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t dim
    ) override {
        // Map to store original indices (handle -> index)
        // Note: We use the handle type from the triangulation instance
        std::map<typename Rt::Vertex_handle, int> handle_map;
        
        Rt rt(static_cast<int>(dim));
        
        for(size_t i=0; i<n_points; ++i) {
             std::vector<double> coords(dim);
             for(size_t d=0; d<dim; ++d) coords[d] = flat_points[i*dim + d];
             Weighted_point_d wp(Point_d(dim, coords.begin(), coords.end()), weights[i]);
             
             auto vh = rt.insert(wp);
             handle_map[vh] = (int)i;
        }

        std::vector<std::pair<int, int>> edges;
        
        // Iterate over finite full cells
        for (auto cit = rt.finite_full_cells_begin(); cit != rt.finite_full_cells_end(); ++cit) {
            // Get current dimension of the cell (should be dim if full dimension)
            int current_dim = rt.current_dimension(); 
            // In dD, a full cell has current_dim + 1 vertices
            
            // Collect vertices
            std::vector<typename Rt::Vertex_handle> vertices;
            vertices.reserve(current_dim + 1);
            for(int k=0; k<=current_dim; ++k) {
                vertices.push_back(cit->vertex(k));
            }
            
            // Generate edges (pairs)
            for(size_t a=0; a<vertices.size(); ++a) {
                for(size_t b=a+1; b<vertices.size(); ++b) {
                    auto v1 = vertices[a];
                    auto v2 = vertices[b];
                    
                    // Filter finite edges
                    // We check if both vertices are in our map (which means they are finite input vertices)
                    if (handle_map.count(v1) && handle_map.count(v2)) {
                        int i1 = handle_map[v1];
                        int i2 = handle_map[v2];
                        if (i1 < i2) edges.push_back({i1, i2});
                        else edges.push_back({i2, i1});
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

    std::vector<std::pair<int, int>> get_standard_delaunay_edges(
        const double* flat_points, 
        size_t n_points,
        size_t dim
    ) override {
        std::map<typename Dt::Vertex_handle, int> handle_map;
        
        Dt dt(static_cast<int>(dim));
        
        for(size_t i=0; i<n_points; ++i) {
             std::vector<double> coords(dim);
             for(size_t d=0; d<dim; ++d) coords[d] = flat_points[i*dim + d];
             Point_d p(dim, coords.begin(), coords.end());
             
             auto vh = dt.insert(p);
             handle_map[vh] = (int)i;
        }

        std::vector<std::pair<int, int>> edges;
        
        for (auto cit = dt.finite_full_cells_begin(); cit != dt.finite_full_cells_end(); ++cit) {
            int current_dim = dt.current_dimension();
            
            std::vector<typename Dt::Vertex_handle> vertices;
            vertices.reserve(current_dim + 1);
            for(int k=0; k<=current_dim; ++k) {
                vertices.push_back(cit->vertex(k));
            }
            
            for(size_t a=0; a<vertices.size(); ++a) {
                for(size_t b=a+1; b<vertices.size(); ++b) {
                    auto v1 = vertices[a];
                    auto v2 = vertices[b];
                    
                    if (handle_map.count(v1) && handle_map.count(v2)) {
                        int i1 = handle_map[v1];
                        int i2 = handle_map[v2];
                        if (i1 < i2) edges.push_back({i1, i2});
                        else edges.push_back({i2, i1});
                    }
                }
            }
        }
        
        std::sort(edges.begin(), edges.end());
        auto last = std::unique(edges.begin(), edges.end());
        edges.erase(last, edges.end());
        
        return edges;
    }

    double compute_simplex_squared_radius(
        const double* /*flat_points*/,
        const int* /*indices*/,
        size_t /*k*/,
        size_t /*dim*/
    ) override {
        // dD Min_sphere compilation issues with Epick_d/Epeck_d traits,
        // and using Cartesian_d<double> can cause infinite loops (hangs) in CGAL.
        // We safely fallback to Python's robust minimum_enclosing_ball calculation.
        return -1.0; 
    }
};

inline std::unique_ptr<WeightedDelaunayTraits> create_cgal_kernel(int dim, bool exact) {
    if (exact) {
        using K = CGAL::Exact_predicates_exact_constructions_kernel;
        if (dim == 2) return std::make_unique<WeightedDelaunay2D<K>>();
        if (dim == 3) return std::make_unique<WeightedDelaunay3D<K>>();
        return std::make_unique<WeightedDelaunayDD<CGAL::Epeck_d<CGAL::Dynamic_dimension_tag>>>();
    } else {
        using K = CGAL::Exact_predicates_inexact_constructions_kernel;
        if (dim == 2) return std::make_unique<WeightedDelaunay2D<K>>();
        if (dim == 3) return std::make_unique<WeightedDelaunay3D<K>>();
        return std::make_unique<WeightedDelaunayDD<CGAL::Epick_d<CGAL::Dynamic_dimension_tag>>>();
    }
}

#endif // HGP_WITH_CGAL