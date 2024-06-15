import pathlib
import librosa
import numpy as np
import scipy
from tqdm import tqdm
import torch
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import datetime
from loguru import logger
import os
import wandb

import itertools

from utils import compute_frequency_representation
from sensory_preprocessor import SensoryPreprocessor
from brain_modules import FFNet
import json

warnings.filterwarnings(action='ignore')
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    print('Using CPU')


recording_name = 'Movie_021'
path_to_audio = pathlib.Path(f'./data/processed/{recording_name}/audio/audio.mp3')
path_to_video = pathlib.Path(f'./data/processed/{recording_name}/frames')
path_to_aux = pathlib.Path(f'./data/trajectories/3000s_dt_0.02_switch_rect_riab_trajectory.json')
audio, sr = librosa.load(path_to_audio,sr=None, mono=False)
frames = [i for i in path_to_video.glob('*.jpg')]
visual_embedding_dim = 200
img_dim = (160, 120)

grayscale=False
c_channels = 1 if grayscale else 3

tsteps = 10
batch_size = 50
n_frames = 150000

train_from_previous = False


sp = SensoryPreprocessor(audio_path=path_to_audio,
                         video_path=path_to_video,
                         aux_path=path_to_aux,
                         tsteps=tsteps,
                         n_frames=n_frames,
                         grayscale=grayscale)
sp.ttv_split(seed=seed)

n_cycles = 50
n_epochs = 20
auditory_loss_fn = torch.nn.MSELoss()

freq_dim = 884
auditory_hidden_dim = 400
auditory_embedding_dim = 200

auditory_enc = FFNet(input_dim=freq_dim, hidden_dim=auditory_hidden_dim, output_dim=auditory_embedding_dim, device=device)
auditory_dec = FFNet(input_dim=auditory_embedding_dim, hidden_dim=auditory_hidden_dim, output_dim=freq_dim)

if train_from_previous is not False:
    auditory_enc.load_state_dict(torch.load(f'./trained_models/{train_from_previous}_auditory_enc_{auditory_hidden_dim}_{auditory_embedding_dim}_{train_from_previous}.pth'))
    auditory_dec.load_state_dict(torch.load(f'./trained_models/{train_from_previous}_auditory_dec_{auditory_hidden_dim}_{auditory_embedding_dim}_{train_from_previous}.pth'))

def save_model():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(auditory_enc.state_dict(), f'./trained_models/{recording_name}_auditory_enc_{auditory_hidden_dim}_{auditory_embedding_dim}_{timestamp}.pth')
    torch.save(auditory_dec.state_dict(), f'./trained_models/{recording_name}_auditory_dec_{auditory_hidden_dim}_{auditory_embedding_dim}_{timestamp}.pth')
    logger.info(f'Saving trained auditory module timestamp: {timestamp}')


for cycle in tqdm(range(n_cycles)):
    vision_params = [auditory_enc.parameters(), auditory_dec.parameters()]
    optim = torch.optim.Adam(itertools.chain(*vision_params), lr=8*1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', 
                                                        factor=0.8, 
                                                        patience=100, 
                                                        threshold=0.01)
    
    # train auditory modules
    auditory_enc.train()
    auditory_dec.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for batch in range(sp.n_frames // (tsteps*batch_size) - 1):
            _ , audio, _ = sp.get_batches(batch_size=batch_size, ttv='tr')
            audio = torch.as_tensor(audio, dtype=torch.float32).view(-1, audio.shape[-1]).to(device)
            audio_recon = auditory_dec(auditory_enc(audio))
            loss = auditory_loss_fn(audio_recon, audio)

            optim.zero_grad()
            loss.backward()

            optim.step()
            epoch_loss += loss.item()
            scheduler.step(loss)
        
        print(f'epoch {epoch+1} auditory loss {np.mean(epoch_loss)}')
        if scheduler.get_last_lr()[0] < 1e-9:
            break
        
    
    # evaluate on test set
    auditory_enc.eval()
    auditory_dec.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in range(sp.n_frames // (tsteps*batch_size) - 1):

            _ , audio, _ = sp.get_batches(batch_size=batch_size, ttv='te')
            audio = torch.as_tensor(audio, dtype=torch.float32).view(-1, audio.shape[-1]).to(device)
            audio_recon = auditory_dec(auditory_enc(audio))
            loss = auditory_loss_fn(audio_recon, audio)
            test_loss += loss.item()

        print(f'cycle {cycle+1} test auditory loss {np.mean(test_loss)}')

    if os.path.exists('./stop_visual_training.txt'):
        save_model()
        break
    
save_model()


