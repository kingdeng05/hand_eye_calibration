cmake_minimum_required(VERSION 3.10)
project(kinetic_backend)

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package
find_package(Eigen3 REQUIRED) 
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# # Add your source files here
# set(SOURCE_FILES src/dhe_factor_graph.cpp)

# Interface library for header-only
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)  # And this line
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(${CMAKE_PROJECT_NAME} INTERFACE)
target_include_directories(kinetic_backend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>  # <prefix>/include
)

# Linking libraries
target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE gtsam ${Boost_LIBRARIES})  

# To install the header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION include
        FILES_MATCHING PATTERN "*.h")

# To install kinetic_factors shared library into system lib directory
install(TARGETS ${CMAKE_PROJECT_NAME}
        EXPORT ${CMAKE_PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib 
        LIBRARY DESTINATION lib 
        RUNTIME DESTINATION lib
        INCLUDES DESTINATION include)
install(EXPORT ${CMAKE_PROJECT_NAME}Targets
        FILE ${CMAKE_PROJECT_NAME}Targets.cmake
        NAMESPACE kinetic_backend::
        DESTINATION lib/cmake/kinetic_backend)

# Pybind11 (if needed)
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})
pybind11_add_module(py_kinetic_backend src/pybind.cpp)
target_link_libraries(py_kinetic_backend PRIVATE gtsam ${CMAKE_PROJECT_NAME})

# To install py_kinetic_factors Python binding into python site-packages
install(TARGETS py_kinetic_backend 
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
