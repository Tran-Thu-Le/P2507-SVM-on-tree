from setuptools import setup, Extension
import sys
try:
    import pybind11
    inc = [pybind11.get_include(), pybind11.get_include(user=True)]
except Exception:
    print("Please: pip install pybind11")
    raise

extra_compile_args = ["-O3"]
if sys.platform == "darwin" or sys.platform.startswith("linux"):
    extra_compile_args = ["-std=c++17", "-O3"]
elif sys.platform.startswith("win"):
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc"]

ext = Extension(
    "svm_on_tree_cpp",
    sources=["src/svm_on_tree.cpp"],
    include_dirs=inc,
    language="c++",
    extra_compile_args=extra_compile_args,
)

setup(
    name="svm_on_tree_cpp",
    version="0.1.0",
    description="SVM On Tree core in C++ (pybind11)",
    ext_modules=[ext],
    zip_safe=False,
)
