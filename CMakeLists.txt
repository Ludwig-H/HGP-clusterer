cmake_minimum_required(VERSION 3.15)
project(hgp_clusterer_extensions LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python explicitly (Fixes Python.h missing in some envs)
find_package(Python3 3.8 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Python3 Found: ${Python3_FOUND}")
message(STATUS "Python3 Include Dirs: ${Python3_INCLUDE_DIRS}")

# Force global include (Safety net)
include_directories(${Python3_INCLUDE_DIRS})
message(STATUS "DEBUG: Forcing include directories: ${Python3_INCLUDE_DIRS}")

# Find Pybind11 (Always Required)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Define the extension module
pybind11_add_module(geometry_binding src/hgp_clusterer/_geometry_binding.cpp)

# Ensure Python headers are found (Critical fix)
if(Python3_FOUND)
    target_link_libraries(geometry_binding PRIVATE Python3::Module)
    target_include_directories(geometry_binding PRIVATE ${Python3_INCLUDE_DIRS})
endif()

# Include directories
target_include_directories(geometry_binding PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/CGALDelaunay/orderk_delaunay_cpp/src
)

# --- DEPENDENCIES ---

# 1. Eigen3 (Required for both CGAL and Geogram backends logic often depends on it, certainly our Welzl implementation)
find_package(Eigen3 QUIET)

if(TARGET Eigen3::Eigen)
    target_link_libraries(geometry_binding PRIVATE Eigen3::Eigen)
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
        target_include_directories(geometry_binding PRIVATE ${EIGEN3_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "Eigen3 not found. Please install libeigen3-dev or set EIGEN3_INCLUDE_DIR.")
    endif()
endif()

# 2. TBB (Optional but recommended for Parallelism)
find_package(TBB QUIET)
if(TARGET TBB::tbb)
    target_link_libraries(geometry_binding PRIVATE TBB::tbb)
    if(TARGET TBB::tbbmalloc)
        target_link_libraries(geometry_binding PRIVATE TBB::tbbmalloc)
    endif()
    # Enable parallelism in _geometry_binding.cpp
    target_compile_definitions(geometry_binding PRIVATE CGAL_LINKED_WITH_TBB) 
    message(STATUS "TBB found! Parallelism enabled.")
endif()

# 3. CGAL Support (Optional)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

find_package(CGAL)

if(NOT CGAL_FOUND)
    message(STATUS "CGAL Config not found. Attempting manual search for headers and libraries...")
    find_path(CGAL_INCLUDE_DIR NAMES CGAL/basic.h
        PATHS 
        $ENV{CGAL_DIR}/include
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
        /opt/local/include
    )
    
    find_library(GMP_LIB gmp PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /lib/x86_64-linux-gnu /lib)
    find_library(MPFR_LIB mpfr PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /lib/x86_64-linux-gnu /lib)
    
    message(STATUS "  - CGAL_INCLUDE_DIR: ${CGAL_INCLUDE_DIR}")
    message(STATUS "  - GMP_LIB: ${GMP_LIB}")
    message(STATUS "  - MPFR_LIB: ${MPFR_LIB}")

    if(CGAL_INCLUDE_DIR AND GMP_LIB AND MPFR_LIB)
        message(STATUS "CGAL found manually at ${CGAL_INCLUDE_DIR}")
        set(CGAL_FOUND TRUE)
        
        # Create imported target manually
        if(NOT TARGET CGAL::CGAL)
            add_library(CGAL::CGAL INTERFACE IMPORTED)
            set_target_properties(CGAL::CGAL PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${CGAL_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "${GMP_LIB};${MPFR_LIB}"
            )
        endif()
    else()
        message(STATUS "Manual fallback failed: Missing one of include dir, GMP, or MPFR.")
    endif()
endif()

if(CGAL_FOUND)
    message(STATUS "CGAL found! Enabled HGP_WITH_CGAL")
    target_compile_definitions(geometry_binding PRIVATE HGP_WITH_CGAL)
    target_link_libraries(geometry_binding PRIVATE CGAL::CGAL)
else()
    message(STATUS "CGAL not found. CGAL backend will be disabled.")
endif()

# 4. Geogram Support (Optional)
if(TRUE)
    if(DEFINED ENV{GEOGRAM_INSTALL_PREFIX})
        list(APPEND CMAKE_PREFIX_PATH $ENV{GEOGRAM_INSTALL_PREFIX})
        message(STATUS "Added Geogram search path from env: $ENV{GEOGRAM_INSTALL_PREFIX}")
    endif()

    find_package(Geogram QUIET)

    if (NOT Geogram_FOUND AND NOT TARGET geogram::geogram)
        # Manual Fallback
        message(STATUS "Geogram config not found. Attempting manual search...")
        
        find_path(GEOGRAM_INCLUDE_DIR NAMES geogram/basic/common.h
            PATHS 
            $ENV{GEOGRAM_INSTALL_PREFIX}/include
            "${CMAKE_CURRENT_SOURCE_DIR}/geogram_install/include"
            /usr/local/include
            /usr/include
            /opt/local/include
            PATH_SUFFIXES geogram1
        )
        
        find_library(GEOGRAM_LIBRARY NAMES geogram
            PATHS
            $ENV{GEOGRAM_INSTALL_PREFIX}/lib
            "${CMAKE_CURRENT_SOURCE_DIR}/geogram_install/lib"
            /usr/local/lib
            /usr/lib
            /opt/local/lib
        )

        if (GEOGRAM_INCLUDE_DIR AND GEOGRAM_LIBRARY)
            message(STATUS "Geogram found manually at: ${GEOGRAM_INCLUDE_DIR}")
            set(Geogram_FOUND TRUE)
            add_library(geogram_imported UNKNOWN IMPORTED)
            set_target_properties(geogram_imported PROPERTIES
                IMPORTED_LOCATION "${GEOGRAM_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GEOGRAM_INCLUDE_DIR}"
            )
            add_library(geogram::geogram ALIAS geogram_imported)
        endif()
    endif()

    # Standardize Geogram usage
    if (TARGET geogram::geogram OR TARGET geogram)
        message(STATUS "Geogram found! Enabled HGP_WITH_GEOGRAM")
        target_compile_definitions(geometry_binding PRIVATE HGP_WITH_GEOGRAM)
        
        if(TARGET geogram::geogram)
            target_link_libraries(geometry_binding PRIVATE geogram::geogram)
        else()
            target_link_libraries(geometry_binding PRIVATE geogram)
        endif()
    else()
        message(STATUS "Geogram not found. Geogram backend will be disabled.")
    endif()
endif()

# Validate that we have at least one backend
if (NOT CGAL_FOUND AND NOT Geogram_FOUND AND NOT TARGET geogram)
    message(FATAL_ERROR "Neither CGAL nor Geogram were found. At least one geometric backend is required to build the extension.")
endif()

# Compiler options
if(MSVC)
    target_compile_options(geometry_binding PRIVATE /O2 /Ob2 /DNDEBUG)
else()
    target_compile_options(geometry_binding PRIVATE -O3 -DNDEBUG)
endif()