cmake_minimum_required(VERSION 3.22)
project(OrderKDelaunay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB QUIET)

add_executable(orderk_delaunay_cpp src/main.cpp)

# Include directories if headers are in src/
target_include_directories(orderk_delaunay_cpp PRIVATE src)

if(TARGET Eigen3::Eigen)
  target_link_libraries(orderk_delaunay_cpp PRIVATE CGAL::CGAL Eigen3::Eigen)
else()
  target_link_libraries(orderk_delaunay_cpp PRIVATE CGAL::CGAL)
  target_include_directories(orderk_delaunay_cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(TARGET TBB::tbb)
    target_link_libraries(orderk_delaunay_cpp PRIVATE TBB::tbb)
    
    if(TARGET TBB::tbbmalloc)
        target_link_libraries(orderk_delaunay_cpp PRIVATE TBB::tbbmalloc)
    else()
        find_library(TBBMALLOC_LIB tbbmalloc)
        if(TBBMALLOC_LIB)
            target_link_libraries(orderk_delaunay_cpp PRIVATE ${TBBMALLOC_LIB})
        endif()
    endif()

    target_compile_definitions(orderk_delaunay_cpp PRIVATE CGAL_LINKED_WITH_TBB)
endif()

target_compile_definitions(orderk_delaunay_cpp PRIVATE HGP_WITH_CGAL)

if(MSVC)
  target_compile_options(orderk_delaunay_cpp PRIVATE /O2 /Ob2 /DNDEBUG)
else()
  target_compile_options(orderk_delaunay_cpp PRIVATE -O3 -march=native -DNDEBUG)
endif()
