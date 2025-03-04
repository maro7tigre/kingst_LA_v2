cmake_minimum_required(VERSION 3.10)
project(kingst_analyzer_cpp)

set(CMAKE_CXX_STANDARD 11)

# Ensure building for 64-bit (to match a 64-bit Python)
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "You must build this project with a 64-bit toolchain to match your 64-bit Python")
endif()

# Find Python3 (change COMPONENTS if you need Interpreter/Development etc.)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add the pybind11 subdirectory (assumes pybind11 is cloned into ./pybind11)
add_subdirectory(pybind11)

# Include directories for Kingst SDK headers and Python headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include)

# Set library search paths for Kingst SDK libraries
if(WIN32)
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/Win64)
else()
    link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/Linux)
endif()

# List your binding source files from src/
set(SOURCES
    src/main.cpp
    src/bind_basic_types.cpp
    src/bind_analyzer.cpp
    src/bind_analyzer_settings.cpp
    src/bind_analyzer_results.cpp
    src/bind_channel_data.cpp
    src/bind_simulation.cpp
    src/bind_helpers.cpp
)

# Create the Python module (the module name can be kingst_analyzer or another name)
pybind11_add_module(kingst_analyzer ${SOURCES})

# Link the module with the Kingst SDK libraries and Python libraries
if(WIN32)
    target_link_libraries(kingst_analyzer PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Win64/Analyzer.lib
        ${Python3_LIBRARIES}
    )
else()
    target_link_libraries(kingst_analyzer PRIVATE 
        Analyzer
        ${Python3_LIBRARIES}
    )
endif()

# Optionally install the built module into the Python package directory
install(TARGETS kingst_analyzer DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer)
