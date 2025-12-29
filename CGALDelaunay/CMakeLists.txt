cmake_minimum_required(VERSION 3.22)
project(EdgesCGALWeightedDelaunay3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Dependencies
find_package(CGAL REQUIRED)
find_package(TBB QUIET)
find_package(Eigen3 REQUIRED)

# TBB Handling
set(USE_TBB OFF)
set(TBB_MALLOC_TARGET "")
if(TBB_FOUND)
  if(TARGET TBB::tbb)
    set(USE_TBB ON)
  endif()
  if(TARGET TBB::tbbmalloc)
    set(TBB_MALLOC_TARGET TBB::tbbmalloc)
  else()
    find_library(TBBMALLOC_LIB tbbmalloc)
    if(TBBMALLOC_LIB)
      set(TBB_MALLOC_TARGET ${TBBMALLOC_LIB})
    endif()
  endif()
  if(USE_TBB AND TBB_MALLOC_TARGET STREQUAL "")
    message(WARNING "TBB found but tbbmalloc missing => disabling parallel CGAL path.")
    set(USE_TBB OFF)
  endif()
endif()

# Common Compilation Flags
function(set_perf_flags target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG -march=native)
    elseif(MSVC)
        target_compile_options(${target_name} PRIVATE /O2 /DNDEBUG)
    endif()
    
    if(USE_TBB)
        target_link_libraries(${target_name} PRIVATE TBB::tbb ${TBB_MALLOC_TARGET})
        target_compile_definitions(${target_name} PRIVATE CGAL_LINKED_WITH_TBB)
    endif()
endfunction()

# -------------------------------------------------------------------------
# Original Tool (Preserved)
# -------------------------------------------------------------------------
add_executable(EdgesCGALWeightedDelaunay3D src/EdgesCGALWeightedDelaunay3D.cpp)
target_link_libraries(EdgesCGALWeightedDelaunay3D PRIVATE CGAL::CGAL)
set_perf_flags(EdgesCGALWeightedDelaunay3D)

# -------------------------------------------------------------------------
# New Order-K Tool
# -------------------------------------------------------------------------
add_executable(OrderKDelaunay src/CGAL_OrderK/main.cpp)
if(TARGET Eigen3::Eigen)
  target_link_libraries(OrderKDelaunay PRIVATE CGAL::CGAL Eigen3::Eigen)
else()
  target_link_libraries(OrderKDelaunay PRIVATE CGAL::CGAL)
  target_include_directories(OrderKDelaunay PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
set_perf_flags(OrderKDelaunay)
# CGAL 5.0+ usually includes Eigen if needed for dD, but dD regular triangulation 
# might require explicit linking if not header-only. Usually header only.