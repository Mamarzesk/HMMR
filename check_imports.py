import numpy as np
import pyminc
import scipy
import sys
import torch
import torch_cubic_spline_grids

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"numpy version: {np.__version__}")
    print(f"pyminc package: {pyminc.__package__}")
    print(f"scipy version: {scipy.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch gpu available: {torch.cuda.is_available()}")
    print(f"torch_cubic_spline_grids version: {torch_cubic_spline_grids.__version__}")
    print("All packages are installed correctly!")
