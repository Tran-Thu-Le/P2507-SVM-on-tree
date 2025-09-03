from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "svm_on_tree_cpp",
        sources=["src/svm_on_tree_lambda_any.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++17"],
    )
]

setup(
    name="svm-on-tree-cpp",
    version="0.3.0",
    description="SVM-on-Tree C++ core (arbitrary lambda) with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
