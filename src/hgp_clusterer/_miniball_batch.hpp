#pragma once
#include "miniball.hpp"
#include <vector>

void compute_miniball_radii_batch(
    const double* M,
    const int* simplex_indices,
    const bool* mask,
    double* radii,
    int n_simplices,
    int K_plus_1,
    int dim
) {
    #pragma omp parallel for
    for (int i = 0; i < n_simplices; ++i) {
        if (mask[i]) {
            std::vector<const double*> points(K_plus_1);
            for (int k = 0; k < K_plus_1; ++k) {
                points[k] = M + simplex_indices[i * K_plus_1 + k] * dim;
            }
            
            typedef const double** PIt;
            typedef const double* CIt;
            typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;
            
            MB mb(dim, points.data(), points.data() + K_plus_1);
            radii[i] = mb.squared_radius();
        }
    }
}

void compute_single_miniball(
    const double* points_flat,
    int n_points,
    int dim,
    double* out_center,
    double* out_radius_sq
) {
    std::vector<const double*> points(n_points);
    for (int k = 0; k < n_points; ++k) {
        points[k] = points_flat + k * dim;
    }
    
    typedef const double** PIt;
    typedef const double* CIt;
    typedef Miniball::Miniball <Miniball::CoordAccessor<PIt, CIt> > MB;
    
    MB mb(dim, points.data(), points.data() + n_points);
    *out_radius_sq = mb.squared_radius();
    for (int i = 0; i < dim; ++i) {
        out_center[i] = mb.center()[i];
    }
}
