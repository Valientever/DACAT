import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the current GPU device (if available)
if torch.cuda.is_available():
    print("Current GPU device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Number of GPUs available:", torch.cuda.device_count())
else:
    print("No GPU detected.")

# Test with a tensor on the GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # Set GPU as the device
    tensor = torch.rand(3, 3).to(device)  # Create a random tensor on the GPU
    print("Tensor on GPU:", tensor)
else:
    print("Running on CPU only.")
