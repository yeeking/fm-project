import torch
import numpy as np
import soundfile as sf
import os 
import sys
import jack
import numpy as np

def encode_decode(model, input):
    latent = model.encode(input)
    # mess with the latent 
    output = model.decode(latent)
    # x = x_hat.cpu().numpy().reshape(-1, 2)  # ensure stereo shape
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


def setup_audio():
    client = jack.Client("white_noise_io")

    # Register ports
    outport = client.outports.register("output")
    inport = client.inports.register("input")

    @client.set_process_callback
    def process(frames):
        # Generate and write white noise to output
        noise = np.random.uniform(-0.2, 0.2, frames).astype(np.float32)
        outport.get_buffer()[:] = noise

        # Read from input, calculate abs mean
        input_data = np.copy(inport.get_buffer())
        abs_mean = np.abs(input_data).mean()
        # print(f"Input abs mean: {abs_mean:.5f}")

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


