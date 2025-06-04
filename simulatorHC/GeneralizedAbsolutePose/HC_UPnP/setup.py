from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import torch

# Function to get MKL include and library paths
def get_mkl_paths():
    mkl_root = os.environ.get('MKLROOT', '')  # Get MKLROOT environment variable
    include_dirs = [os.path.join(mkl_root, 'include')]
    library_dirs = [os.path.join(mkl_root, 'lib', 'intel64')]
    return include_dirs, library_dirs

# Retrieve MKL paths
mkl_include_dirs, mkl_library_dirs = get_mkl_paths()

# Define your CppExtension with MKL include and library directories
ext_modules = [
    CppExtension(
        'HC_UPnP',
        ['upnp_eigen3.cpp', 'utils.cpp'],
        include_dirs=[
            '/usr/include/eigen3',
            *torch.utils.cpp_extension.include_paths(),
            *mkl_include_dirs  # Add MKL include directory
        ],
        library_dirs=[
            *torch.utils.cpp_extension.library_paths(),
            *mkl_library_dirs  # Add MKL library directory
        ],
        libraries=['c10', 'torch', 'torch_cpu', 'torch_python', 'mkl_rt'],  # Add 'mkl_rt' for MKL runtime library
        extra_compile_args=['-O3', '-march=native', '-flto', '-mssse3'],
    )
]

setup(
    name='HC_UPnP',
    version='0.0.1',
    packages=find_packages(),
    description='homotopy continuation solver for Generalized Absolute Pose problem (UPnP)',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    }
)
