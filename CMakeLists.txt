cmake_minimum_required(VERSION 3.10)
project(kingst_analyzer_cpp)

set(CMAKE_CXX_STANDARD 11)

# Ensure building for 64-bit (to match a 64-bit Python)
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "You must build this project with a 64-bit toolchain to match your 64-bit Python")
endif()

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add the pybind11 subdirectory
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

# Create the Python module - named _core
pybind11_add_module(_core ${SOURCES})

# Link the module with the Kingst SDK libraries and Python libraries
if(WIN32)
    target_link_libraries(_core PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Win64/Analyzer.lib
        ${Python3_LIBRARIES}
    )
    
    # Set RUNTIME_OUTPUT_DIRECTORY to put the .pyd file directly in the package
    set_target_properties(_core PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer
    )
    
    # Add a post-build command to copy the required DLLs to the package directory
    add_custom_command(TARGET _core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/Win64/Analyzer.dll
            ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer/
        COMMAND ${CMAKE_COMMAND} -E echo "Copying Analyzer.dll to package directory"
    )
    
    # Add any other DLLs that might be needed
    # For example, if you have other DLLs, add them here:
    # COMMAND ${CMAKE_COMMAND} -E copy
    #     ${CMAKE_CURRENT_SOURCE_DIR}/lib/Win64/OtherDLL.dll
    #     ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer/
else()
    target_link_libraries(_core PRIVATE 
        Analyzer
        ${Python3_LIBRARIES}
    )
    
    # Set RPATH for Linux/macOS
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer
        INSTALL_RPATH "$ORIGIN"
    )
    
    # Add a post-build command to copy the required shared libraries to the package directory
    add_custom_command(TARGET _core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/lib/Linux/libAnalyzer.so
            ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer/
        COMMAND ${CMAKE_COMMAND} -E echo "Copying libAnalyzer.so to package directory"
    )
endif()

# Installation targets (optional)
install(TARGETS _core DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/kingst_analyzer)