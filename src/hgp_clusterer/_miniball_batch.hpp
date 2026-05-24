#pragma once
#include "miniball.hpp"
#include "kernels_geogram.hpp"
#include <vector>

void compute_miniball_radii_batch(
    const double* M,
    const int* simplex_indices,
    const unsigned char* mask,
    double* radii,
    size_t n_simplices,
    size_t K_plus_1,
    size_t dim
) {
    #pragma omp parallel for
    for (size_t i = 0; i < n_simplices; ++i) {
        if (mask[i]) {
            radii[i] = HGP_Numerics::compute_meb_sq_radius(M, &simplex_indices[i * K_plus_1], K_plus_1, dim);
        }
    }
}

void compute_single_miniball(
    const double* points_flat,
    size_t n_points,
    size_t dim,
    double* out_center,
    double* out_radius_sq
) {
    std::vector<const double*> points(n_points);
    for (size_t k = 0; k < n_points; ++k) {
        points[k] = points_flat + k * dim;
    }
    
    typedef const double** PIt;
    typedef const double* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;
    
    MB mb(dim, points.data(), points.data() + n_points);
    *out_radius_sq = mb.squared_radius();
    for (size_t i = 0; i < dim; ++i) {
        out_center[i] = mb.center()[i];
    }
}
