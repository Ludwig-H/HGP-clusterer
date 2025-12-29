#pragma once

#include <vector>
#include <utility>
#include <iostream>
#include <map>

// CGAL Headers
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_vertex_base_2.h>
#include <CGAL/Regular_triangulation_face_base_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

// dD Headers (Partial support)
#include <CGAL/Epick_d.h>
#include <CGAL/Epeck_d.h>
//#include <CGAL/Regular_triangulation.h> 
//#include <CGAL/Regular_triangulation_traits_adapter.h>
//#include <CGAL/Triangulation_vertex.h>
//#include <CGAL/Triangulation_data_structure.h>

// TBB
#ifdef CGAL_LINKED_WITH_TBB
#include <CGAL/Triangulation_data_structure_3.h>
#endif

// Interface for weighted Delaunay edge extraction
struct WeightedDelaunayTraits {
    virtual std::vector<std::pair<int, int>> get_finite_edges(
        const std::vector<double>& flat_points, 
        const std::vector<double>& weights,
        size_t n_points,
        size_t dim
    ) = 0;
    virtual ~WeightedDelaunayTraits() {}
};

// 2D Implementation
template <typename Kernel>
struct WeightedDelaunay2D : public WeightedDelaunayTraits {
    using Vb0 = CGAL::Regular_triangulation_vertex_base_2<Kernel>;
    using Vb  = CGAL::Triangulation_vertex_base_with_info_2<int, Kernel, Vb0>;
    using Fb  = CGAL::Regular_triangulation_face_base_2<Kernel>;
    using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
    using Rt  = CGAL::Regular_triangulation_2<Kernel, Tds>;
    using Weighted_point = typename Kernel::Weighted_point_2;
    using Point_2 = typename Kernel::Point_2;

    std::vector<std::pair<int, int>> get_finite_edges(
        const std::vector<double>& flat_points, 
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
};

// 3D Implementation
template <typename Kernel>
struct WeightedDelaunay3D : public WeightedDelaunayTraits {
    using Vb0 = CGAL::Regular_triangulation_vertex_base_3<Kernel>;
    using Vb  = CGAL::Triangulation_vertex_base_with_info_3<int, Kernel, Vb0>;
    using Cb  = CGAL::Regular_triangulation_cell_base_3<Kernel>;
    #ifdef CGAL_LINKED_WITH_TBB
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag>;
    #else
    using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
    #endif
    using Rt  = CGAL::Regular_triangulation_3<Kernel, Tds>;
    using Weighted_point = typename Kernel::Weighted_point_3;
    using Point_3 = typename Kernel::Point_3;

    std::vector<std::pair<int, int>> get_finite_edges(
        const std::vector<double>& flat_points, 
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
};

// dD Implementation (Stub)
template <typename Kernel>
struct WeightedDelaunayDD : public WeightedDelaunayTraits {
    std::vector<std::pair<int, int>> get_finite_edges(
        const std::vector<double>&, 
        const std::vector<double>&,
        size_t,
        size_t
    ) override {
        std::cerr << "[Error] dD Regular Triangulation not fully supported in this build due to compilation issues.\n";
        return {};
    }
};

std::unique_ptr<WeightedDelaunayTraits> create_kernel(int dim, bool exact) {
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