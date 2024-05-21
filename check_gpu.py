import torch

# Check if GPU is available

if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPU is not available. Please check your installation.")
# Try to print CUDA version and device properties
try:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch built with CUDA: {torch.backends.cudnn.enabled}")
except Exception as e:
    print(f"Error checking CUDA version or properties: {e}")
