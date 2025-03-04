from setuptools import setup, Extension
import sys
import os

# Try to import pybind11 with helpful error message if it fails
try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("ERROR: pybind11 is required. Install it with:")
    print("    pip install pybind11")
    sys.exit(1)

# Get pybind11 include directory
pybind11_include = pybind11.get_include()
print(f"Using pybind11 from: {pybind11_include}")

# Provide correct name of Kingst library - adjust as needed
kingst_lib_name = "KingstLA"  # Or whatever the actual library name is

# List of source files
sources = [
    "src/main.cpp",
    "src/bind_basic_types.cpp",
    "src/bind_analyzer.cpp",
    "src/bind_analyzer_settings.cpp",
    "src/bind_analyzer_results.cpp",
    "src/bind_channel_data.cpp",
    "src/bind_helpers.cpp",
    "src/bind_simulation.cpp",
]

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "kingst_analyzer._core",
        sources=sources,
        include_dirs=[
            "include",
            pybind11_include,
        ],
        library_dirs=["lib"],
        libraries=[kingst_lib_name],
        # Use C++14 which is better supported by VS 2022
        extra_compile_args=["/std:c++14", "/EHsc"],
        language="c++",
    ),
]

setup(
    name="kingst_analyzer",
    version="0.2.0",
    author="maro7tiger",
    description="Python bindings for Kingst Logic Analyzer",
    ext_modules=ext_modules,
    packages=["kingst_analyzer"],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)