cmake_minimum_required(VERSION 3.15)
project(hgp_clusterer_extensions LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Find CGAL and dependencies
find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB QUIET)

# Define the extension module
# The name must match the one in PYBIND11_MODULE(cgal_binding, m) -> "cgal_binding"
# However, we want it to be importable as hgp_clusterer.cgal_binding
# setup.py will handle the install location.
pybind11_add_module(cgal_binding src/hgp_clusterer/_cgal_binding.cpp)

# Include directories
# We need kernels.hpp which is in CGALDelaunay/orderk_delaunay_cpp/src/
target_include_directories(cgal_binding PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/CGALDelaunay/orderk_delaunay_cpp/src
)

# Link libraries
target_link_libraries(cgal_binding PRIVATE CGAL::CGAL)

if(TARGET Eigen3::Eigen)
    target_link_libraries(cgal_binding PRIVATE Eigen3::Eigen)
else()
    target_include_directories(cgal_binding PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(TARGET TBB::tbb)
    target_link_libraries(cgal_binding PRIVATE TBB::tbb)
    if(TARGET TBB::tbbmalloc)
        target_link_libraries(cgal_binding PRIVATE TBB::tbbmalloc)
    endif()
    target_compile_definitions(cgal_binding PRIVATE CGAL_LINKED_WITH_TBB)
endif()

# Compiler options
if(MSVC)
    target_compile_options(cgal_binding PRIVATE /O2 /Ob2 /DNDEBUG)
else()
    target_compile_options(cgal_binding PRIVATE -O3 -march=native -DNDEBUG)
endif()
