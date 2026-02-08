cmake_minimum_required(VERSION 3.15)
project(hgp_clusterer_extensions LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Pybind11 (Always Required)
find_package(pybind11 REQUIRED)

# Define the extension module
pybind11_add_module(cgal_binding src/hgp_clusterer/_cgal_binding.cpp)

# Include directories
target_include_directories(cgal_binding PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/CGALDelaunay/orderk_delaunay_cpp/src
)

# --- DEPENDENCIES ---

# 1. Eigen3 (Required for both CGAL and Geogram backends logic often depends on it, certainly our Welzl implementation)
find_package(Eigen3 QUIET)

if(TARGET Eigen3::Eigen)
    target_link_libraries(cgal_binding PRIVATE Eigen3::Eigen)
else()
    # Try finding via include dir variable if target missing
    message(STATUS "Eigen3 config not found. Searching for headers...")
    find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
        PATHS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /opt/homebrew/include/eigen3
    )
    
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
        target_include_directories(cgal_binding PRIVATE ${EIGEN3_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Eigen3 not found. Please install libeigen3-dev or set EIGEN3_INCLUDE_DIR.")
    endif()
endif()

# 2. TBB (Optional but recommended for CGAL)
find_package(TBB QUIET)

# 3. CGAL Support (Optional)
find_package(CGAL QUIET)
if(CGAL_FOUND)
    message(STATUS "CGAL found! Enabled HGP_WITH_CGAL")
    target_compile_definitions(cgal_binding PRIVATE HGP_WITH_CGAL)
    target_link_libraries(cgal_binding PRIVATE CGAL::CGAL)
    
    if(TARGET TBB::tbb)
        target_link_libraries(cgal_binding PRIVATE TBB::tbb)
        if(TARGET TBB::tbbmalloc)
            target_link_libraries(cgal_binding PRIVATE TBB::tbbmalloc)
        endif()
        target_compile_definitions(cgal_binding PRIVATE CGAL_LINKED_WITH_TBB)
    endif()
else()
    message(STATUS "CGAL not found. CGAL backend will be disabled.")
endif()

# 4. Geogram Support (Optional)
if(DEFINED ENV{GEOGRAM_INSTALL_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH $ENV{GEOGRAM_INSTALL_PREFIX})
    message(STATUS "Added Geogram search path from env: $ENV{GEOGRAM_INSTALL_PREFIX}")
endif()

find_package(Geogram QUIET)
# Standardize Geogram detection (Config mode vs Module mode)
if (Geogram_FOUND OR TARGET geogram OR TARGET geogram::geogram)
    message(STATUS "Geogram found! Enabled HGP_WITH_GEOGRAM")
    target_compile_definitions(cgal_binding PRIVATE HGP_WITH_GEOGRAM)
    
    if (TARGET geogram)
        target_link_libraries(cgal_binding PRIVATE geogram)
    elseif(TARGET geogram::geogram)
        target_link_libraries(cgal_binding PRIVATE geogram::geogram)
    else()
        target_include_directories(cgal_binding PRIVATE ${GEOGRAM_INCLUDE_DIR})
        target_link_libraries(cgal_binding PRIVATE ${GEOGRAM_LIBRARIES})
    endif()
else()
    message(STATUS "Geogram not found. Geogram backend will be disabled.")
endif()

# Validate that we have at least one backend
if (NOT CGAL_FOUND AND NOT Geogram_FOUND AND NOT TARGET geogram)
    message(FATAL_ERROR "Neither CGAL nor Geogram were found. At least one geometric backend is required to build the extension.")
endif()

# Compiler options
if(MSVC)
    target_compile_options(cgal_binding PRIVATE /O2 /Ob2 /DNDEBUG)
else()
    target_compile_options(cgal_binding PRIVATE -O3 -march=native -DNDEBUG)
endif()