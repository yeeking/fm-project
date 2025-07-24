import torch
import numpy as np
import soundfile as sf
import os 
import sys
import jack
import numpy as np
import time
import threading 

import signal 

def encode_decode(model, input_chunk:np.array):
    # Assumes input_chunk is 1D np.float32 array of length N    
    x = torch.from_numpy(input_chunk).float().unsqueeze(0).unsqueeze(0)  # shape [1, 1, N]
    x = x.expand(1, 2, -1)  # shape [1, 2, N] -> stereo via duplication
    print(f"Input shape is {x.shape}")
    with torch.no_grad():
        # latent = model.encode(x)
        # print(f"Latent share {latent.shape}")
        # Optionally modify latent here
        # x_hat = model.decode(latent)
        x_hat = model.forward(x)
        

    output = x_hat.squeeze().cpu().numpy()  # shape [N]
    print(f"encode decode done.")
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



device = setup_device()
model = setup_model(device)
latent_shape = get_latent_shape(model, device)

# Create the JACK client
client = jack.Client("noise_sender")
# Register a single output port
jack_out = client.outports.register("output")
jack_in = client.inports.register("input")
# Auto-connect output to system playback, input to system capture

stop = threading.Event()
signal.signal(signal.SIGINT, lambda *args: stop.set())

rave_block_size=8192

@client.set_process_callback
def process(frames):
    # this bit runs once the script is trying to stop
    # it prevents any further calls to the torch layer which can cause blocks on exit
    if stop.is_set():
        jack_out.get_array()[:] = 0
        return 0
    # normal mode: call out to torch 
    in_buf = jack_in.get_array()
    if in_buf.shape[0] < rave_block_size:
        padded = np.zeros(rave_block_size, dtype=np.float32)
        padded[:frames] = in_buf
        in_buf = padded 
    output_data = encode_decode(model, in_buf)
    buf = jack_out.get_array()  # NumPy array view of the buffer
    buf[:] = (np.random.rand(frames) * 2 - 1).astype(np.float32) * 0.1

    # out.get_buffer()[:] = output_data[0][:frames]


# Thread-safe signal to stop
stop = threading.Event()
signal.signal(signal.SIGINT, lambda *args: stop.set())

client.activate()
playback_ports = client.get_ports(is_physical=True, is_input=True)
capture_ports = client.get_ports(is_physical=True, is_output=True)
if playback_ports:
    jack_out.connect(playback_ports[0])
if capture_ports:
    jack_in.connect(capture_ports[0])

print(f"Sending noise to '{client.name}:output' (fs={client.samplerate} Hz). Press Ctrl+C to quit.")


try:
    while not stop.is_set():
        time.sleep(0.1)
finally:
    time.sleep(0.5) # wait for any existing call to encode_decode to exit
    client.deactivate()
    client.close()
    print("Clean shutdown.")


