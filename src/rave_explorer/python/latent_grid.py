import torch
import itertools
import numpy as np
import soundfile as sf
import os 
import sys
import torch
import itertools

def generate_latent_grid(dim=4, n_steps=5, value_range=(-1.0, 1.0), device="cpu"):
    """
    Create a single latent tensor for decoding: shape [1, 8, N]
    where N = n_steps**dim, duplicating a 4D grid into stereo (8 channels).
    """
    coords = [torch.linspace(value_range[0], value_range[1], n_steps, device=device)
              for _ in range(dim)]
    points = list(itertools.product(*coords))
    N = len(points)  # e.g., 5**4 = 625

    z = torch.zeros(1, dim * 2, N, device=device)
    for i, point in enumerate(points):
        v = torch.tensor(point, dtype=torch.float, device=device)
        z[0, :dim, i] = v
        z[0, dim:, i] = v  # stereo duplication

    return z  # shape [1, 8, 625]


# Load your trained RAVE model (as a scripted Torch model)
assert len(sys.argv) == 2, f"Usage: python script.py <ts export file>"

model_file = sys.argv[1]

assert os.path.exists(model_file), f"Cannot find file {model_file}"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = torch.jit.load(model_file).eval().to(device)  # or .cpu()

# Determine latent shape by passing a dummy input through the encoder
# Note: skip this if you already know latent shape, e.g. (B, C, T)
dummy = torch.zeros(1, 2, 44100 * 1)  # 1 second stereo
z = model.encode(dummy.to(device))
B, C, T = z.shape
print(f'Model latent shape {z.shape} with {C} control dimensions')

z = generate_latent_grid(4,15)

print(f" grid shape is {z.shape}")

# Decode latent to waveform
with torch.no_grad():
    x_hat = model.decode(z)

# Save to WAV
x = x_hat.cpu().numpy().reshape(-1, 2)  # ensure stereo shape
sf.write("generated.wav", x, 44100)
print("🎶 Generated audio saved to generated.wav")

