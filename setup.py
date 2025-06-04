from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import torch

# Function to get MKL include and library paths
def get_mkl_paths():
    mkl_root = os.environ.get('MKLROOT', '')
    if not mkl_root:
        print("Warning: MKLROOT environment variable not set. MKL features might not be available or build might fail.")
        return [], []
    include_dirs = [os.path.join(mkl_root, 'include')]
    # Common MKL library paths, adjust if yours is different (e.g., lib/intel64, lib/x64, or just lib)
    library_dirs = [
        os.path.join(mkl_root, 'lib', 'intel64'), # Common for Linux
        os.path.join(mkl_root, 'lib')            # Sometimes it's directly in lib
    ]
    # Filter out non-existent paths
    library_dirs = [d for d in library_dirs if os.path.isdir(d)]
    if not library_dirs and mkl_root: # if mkl_root was set but intel64 subdir not found
        print(f"Warning: MKL library directory like 'lib/intel64' not found under MKLROOT={mkl_root}. Trying MKLROOT/lib.")

    return include_dirs, library_dirs

mkl_include_dirs, mkl_library_dirs = get_mkl_paths()

# --- Define C++ Extensions ---

# Extension for Problem UPnP
ext_upnp = CppExtension(
    name='simulatorHC.GeneralizedAbsolutePose.HC_UPnP', # <--- CRITICAL: Full Python path to the module
    sources=[
        'simulatorHC/GeneralizedAbsolutePose/HC_UPnP/upnp_eigen3.cpp',
        'simulatorHC/GeneralizedAbsolutePose/HC_UPnP/utils.cpp', # Assumes utils.cpp is in problem_upnp/HC_CPP
    ],
    include_dirs=[
        '/usr/include/eigen3',
        'simulatorHC/GeneralizedAbsolutePose/HC_UPnP/', # For local headers in this extension
        *torch.utils.cpp_extension.include_paths(),
        *mkl_include_dirs
    ],
    library_dirs=[
        *torch.utils.cpp_extension.library_paths(),
        *mkl_library_dirs
    ],
    libraries=['c10', 'torch', 'torch_cpu', 'torch_python', 'mkl_rt'],
    extra_compile_args=['-O3', '-march=native', '-flto', '-mssse3', '-std=c++17'],
)

# Extension for Problem GRPS
ext_grps = CppExtension(
    name='simulatorHC.GeneralizedRelativePoseAndScale.HC_GRPS',  # <--- CRITICAL: Full Python path to the module
    sources=[
        'simulatorHC/GeneralizedRelativePoseAndScale/HC_GRPS/pyadapter.cpp',
        'simulatorHC/GeneralizedRelativePoseAndScale/HC_GRPS/utils.cpp',
    ],
    include_dirs=[
        '/usr/include/eigen3', # Consider making this configurable or checking existence
        'simulatorHC/GeneralizedRelativePoseAndScale/HC_GRPS/', # For local headers in this extension
        *torch.utils.cpp_extension.include_paths(),
        *mkl_include_dirs
    ],
    library_dirs=[
        *torch.utils.cpp_extension.library_paths(),
        *mkl_library_dirs
    ],
    libraries=['c10', 'torch', 'torch_cpu', 'torch_python', 'mkl_rt'],
    extra_compile_args=['-O3', '-march=native', '-flto', '-mssse3', '-std=c++17'], # Added -std=c++17 for good measure
)


setup(
    name='simulatorHC',
    description='Homotopy continuation solvers for pose estimation problems (GRPS and UPnP).',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(where='.'),
    package_dir={'simulatorHC': 'simulatorHC'},
    package_data={'simulatorHC': ['GeneralizedAbsolutePose/HC_UPnP/*', 'GeneralizedRelativePoseAndScale/HC_GRPS/*']},
    
    ext_modules=[ext_grps, ext_upnp], # List of all C++ extensions
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    python_requires='>=3.7', # Example, adjust as needed
)