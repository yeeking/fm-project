import torch
import numpy as np
import soundfile as sf
import os 
import sys
import jack
import numpy as np

import torch

def encode_decode(model, input_chunk:np.array):
    # Assumes input_chunk is 1D np.float32 array of length N    
    x = torch.from_numpy(input_chunk).float().unsqueeze(0).unsqueeze(0)  # shape [1, 1, N]
    x = x.expand(1, 2, -1)  # shape [1, 2, N] -> stereo via duplication

    with torch.no_grad():
        latent = model.encode(x)
        # Optionally modify latent here
        x_hat = model.decode(latent)

    output = x_hat.squeeze().cpu().numpy()  # shape [N]
    return output

    
def setup_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device 

def setup_model(device):
    # Load your trained RAVE model (as a scripted Torch model)
    assert len(sys.argv) == 2, f"Usage: python script.py <ts export file>"

    model_file = sys.argv[1]

    assert os.path.exists(model_file), f"Cannot find file {model_file}"
    print(f"Loading model {model_file} to device {device}")
    model = torch.jit.load(model_file).eval().to(device)  # or .cpu()
    return model 

def get_latent_shape(model, device):
    dummy = torch.zeros(1, 2, 44100 * 1)  # 1 second stereo
    z = model.encode(dummy.to(device))
    B, C, T = z.shape
    print(f'Model latent shape {z.shape} with {C} control dimensions')
    
    z = torch.randn(B, C, 2048, device=device)


def setup_audio(rave_block_size=8192):
    client = jack.Client("white_noise_io")


    # Register ports
    outport = client.outports.register("output")
    inport = client.inports.register("input")

    @client.set_process_callback
    def process(frames):
        input_data = np.frombuffer(inport.get_buffer(), dtype=np.float32).copy()

        # print(f"{input_data}")
        # print(f"Tpye of input data {type(input_data)} len {input_data.shape}")
        # # Optional: buffer up frames until we have enough
        if input_data.shape[0] < rave_block_size:
            # zero-pad or maintain a ring buffer, here's a quick zero-pad version:
            padded = np.zeros(rave_block_size, dtype=np.float32)
            # print(f"Padded: {len(padded)} input: {len(input_data)} frames {frames}")
            padded[:frames] = input_data
            input_data = padded

        output_data = encode_decode(model, input_data)
        # print(f"Received output from model {output_data.shape} out buffer shape {np.frombuffer(outport.get_buffer(), dtype=np.float32).shape}")
        # # Crop if model output is longer than JACK buffer
        # out = output_data[:frames] if len(output_data) >= frames else np.pad(output_data, (0, frames - len(output_data)))
        # print(f"Out shape {out.shape}")

        outport.get_buffer()[:] = output_data[0][:frames]


    @client.set_shutdown_callback
    def shutdown(status, reason):
        print(f"JACK shutdown: {reason}")
        exit(1)
    return client, inport, outport

device = setup_device()
model = setup_model(device)
latent_shape = get_latent_shape(model, device)
client, inport, outport = setup_audio()
# Activate client and connect ports

with client:
    try:
        # Auto-connect output to system playback, input to system capture
        playback_ports = client.get_ports(is_physical=True, is_input=True)
        capture_ports = client.get_ports(is_physical=True, is_output=True)
        if playback_ports:
            outport.connect(playback_ports[0])
        if capture_ports:
            inport.connect(capture_ports[0])

        print("Running. Press Ctrl+C to stop.")
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")



# Determine latent shape by passing a dummy input through the encoder
# Note: skip this if you already know latent shape, e.g. (B, C, T)

# Decode latent to waveform
# with torch.no_grad():


