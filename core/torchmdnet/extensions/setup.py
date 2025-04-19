import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# --- Configuration ---
extension_name = 'torchmdnet_extensions' # Keep this consistent

# --- Source Files ---
# Files that are always compiled (C++ binding and CPU implementation)
cpp_source_files = [
    'extensions.cpp',
    'neighbors/neighbors_cpu.cpp' # Include path relative to setup.py
]

# CUDA specific source files (.cu files containing kernels)
cuda_source_files = [
    'neighbors/neighbors_cuda.cu' # Include path relative to setup.py
]

# Check for CUDA availability
use_cuda = torch.cuda.is_available()
# use_cuda = False # Uncomment to force CPU-only build

# --- Prepare sources and select Extension type ---
sources = cpp_source_files # Start with common C++ files
extra_compile_args = {'cxx': []} # Default C++ compile args

if use_cuda:
    print(f"Compiling {extension_name} with CUDA support.")
    Extension = CUDAExtension
    sources += cuda_source_files # Add CUDA sources if using CUDA
    extra_compile_args['nvcc'] = [] # Add placeholder for NVCC args
    # Example NVCC args: extra_compile_args['nvcc'] = ['-O3', '--use_fast_math']
else:
    print(f"Compiling {extension_name} for CPU only.")
    Extension = CppExtension
    # Optional: Define a macro if your C++ code needs to know it's CPU-only
    # extra_compile_args['cxx'].append('-DCPU_ONLY')

# --- Setup ---
setup(
    name=extension_name, # Package name
    version='0.1',      # Version
    ext_modules=[
        Extension(
            name=extension_name,             # Name of the generated .so file
            sources=sources,                 # Use the dynamically built sources list
            extra_compile_args=extra_compile_args
            # Optional: Explicitly add include directory if needed, but often automatic
            # include_dirs=['neighbors']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension # Use PyTorch's build system
    }
)
