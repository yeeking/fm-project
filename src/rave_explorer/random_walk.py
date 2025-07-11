import torch
import numpy as np
import soundfile as sf
import os 
import sys

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

# Just set your dimensions directly if known
# C = 128       # latent channels (check your model config)
# T = 2048     # number of time steps
# B = 1        # batch size

# Generate random latent vectors or define your own sequence
z = torch.randn(B, C, 2048, device=device)

# Decode latent to waveform
with torch.no_grad():
    x_hat = model.decode(z)

# Save to WAV
x = x_hat.cpu().numpy().reshape(-1, 2)  # ensure stereo shape
sf.write("generated.wav", x, 44100)
print("🎶 Generated audio saved to generated.wav")
