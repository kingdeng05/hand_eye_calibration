cmake_minimum_required(VERSION 3.10)
project(kinetic_backend)

# Find GTSAM components
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Specify the source files to be compiled
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

# Build a shared or static library from source files
add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCE_FILES}) # Use STATIC instead of SHARED for a static library

# Specify the directories that the compiler should look for include files
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> # <prefix>/include
)

# Link the library against GTSAM and Boost libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC gtsam ${Boost_LIBRARIES})

# Install header files, library and CMake targets
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include FILES_MATCHING PATTERN "*.h")
install(TARGETS ${CMAKE_PROJECT_NAME}
        EXPORT ${CMAKE_PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin # usually bin for executables and lib for libraries
        INCLUDES DESTINATION include)
install(EXPORT ${CMAKE_PROJECT_NAME}Targets
        FILE ${CMAKE_PROJECT_NAME}Targets.cmake
        NAMESPACE kinetic_backend::
        DESTINATION lib/cmake/kinetic_backend)

# Pybind11 (if needed)
find_package(pybind11 REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})
pybind11_add_module(py_kinetic_backend src/pybind.cpp)
target_link_libraries(py_kinetic_backend PRIVATE gtsam ${CMAKE_PROJECT_NAME} ${PYTHON_INCLUDE_DIRS})

# To install py_kinetic_factors Python binding into python site-packages
install(TARGETS py_kinetic_backend 
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
