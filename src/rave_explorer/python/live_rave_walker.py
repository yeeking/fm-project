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

from enum import Enum, auto

class RaveMode(Enum):
    WALK = auto()
    LIVE = auto()

def encode_decode(model, input_chunk:np.array, mode:RaveMode, latent_vec):
    # Assumes input_chunk is 1D np.float32 array of length N    
    
    # print(f"Input shape is {x.shape}")
    with torch.no_grad():
        if mode == RaveMode.LIVE:
            x = torch.from_numpy(input_chunk).float().unsqueeze(0).unsqueeze(0)  # shape [1, 1, N]
            x = x.expand(1, 2, -1)  # shape [1, 2, N] -> stereo via duplication
            latent = model.encode(x)
        if mode == RaveMode.WALK:
            # Latent shape torch.Size([1, 8, 4])
            assert latent_vec.shape == (1,8,4), f"Want shape (1,8,4) but got {latent_vec.shape}"
            latent = latent_vec
        # Optionally modify latent here
        latent = latent + np.random.random()
        x_hat = model.decode(latent)
        # x_hat = model.forward(x)
        

    output = x_hat.squeeze().cpu().numpy()  # shape [N]
    # print(f"encode decode done.")
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

# def get_latent_shape(model, device):
#     dummy = torch.zeros(1, 2, 44100 * 1)  # 1 second stereo
#     z = model.encode(dummy.to(device))
#     B, C, T = z.shape
#     print(f'Model latent shape {z.shape} with {C} control dimensions')
    
#     z = torch.randn(B, C, 2048, device=device)


def init_jack():
#  Create the JACK client
    client = jack.Client("rave_io")
    # Register a single output port
    jack_out_L = client.outports.register("outputL")
    jack_out_R = client.outports.register("outputR")
    jack_out = client.outports.register("output")
    jack_in = client.inports.register("input")

    @client.set_process_callback
    def process(frames):
        global rave_buff_pos, rave_input_buff, rave_output_buff, random_latents, latent_pos, RaveMode

        in_buf = jack_in.get_array()
        out_buf = jack_out.get_array()  # NumPy array view of the buffer
        
        # this bit runs once the script is trying to stop
        # it prevents any further calls to the torch layer which can cause blocks on exit
        if stop.is_set():
            jack_out.get_array()[:] = 0
            return 0
        # normal mode: call out to torch 
        # copy latest into input buffer
        rave_input_buff[rave_buff_pos : rave_buff_pos + in_buf.size] = in_buf
        # copy from output 
        # buf[:] = (np.random.rand(frames) * 2 - 1).astype(np.float32) * 0.1
        # print(f"Output buffer shape is {rave_output_buff.shape}")
        out_buf[:] = rave_output_buff[0][rave_buff_pos:rave_buff_pos + in_buf.size]
        
        rave_buff_pos = (rave_buff_pos + in_buf.size) % rave_block_size
        if rave_buff_pos == 0: 
            print(f"Latent is {latent_pos}: ready to infer")
            # ready to process
            rave_output_buff =   encode_decode(model, rave_input_buff, mode=RaveMode.WALK, latent_vec=random_latents[latent_pos].unsqueeze(0))
            
            latent_pos = (latent_pos + 1) % random_latents.shape[0]
    
    return client, jack_in, jack_out

def wire_jack(client:jack.Client, jack_in, jack_out):
    playback_ports = client.get_ports(is_physical=True, is_input=True)
    capture_ports = client.get_ports(is_physical=True, is_output=True)
    
    if playback_ports:
        jack_out.connect(playback_ports[0])
        jack_out.connect(playback_ports[1])
        
    if capture_ports:
        jack_in.connect(capture_ports[0])
        


torch.set_num_threads(1)

# Thread-safe signal to stop
stop = threading.Event()
signal.signal(signal.SIGINT, lambda *args: stop.set())

device = setup_device()
model = setup_model(device)
# latent_shape = get_latent_shape(model, device)


random_latents = torch.rand(5, 8, 4, dtype=torch.float32, device=device)

latent_pos = 0

rave_block_size=8192
rave_input_buff = np.zeros(rave_block_size)
rave_output_buff = np.zeros((2, rave_block_size))
rave_buff_pos = 0

client, jack_in, jack_out = init_jack()

print(f"Activating client {rave_buff_pos}")
client.activate()

wire_jack(client, jack_in, jack_out)

print(f"Sending RAVE signal to '{client.name}:output' (fs={client.samplerate} Hz). Press Ctrl+C to quit.")

try:
    while not stop.is_set():
        time.sleep(0.1)
finally:
    time.sleep(0.5) # wait for any existing call to encode_decode to exit
    client.deactivate()
    client.close()
    print("Clean shutdown.")


