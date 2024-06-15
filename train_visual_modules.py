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
batch_size = 30
n_frames = 150000

train_from_previous = False

from brain_modules import FFVisualModuleEncoder, FFVisualModuleDecoder


sp = SensoryPreprocessor(audio_path=path_to_audio,
                         video_path=path_to_video,
                         aux_path=path_to_aux,
                         tsteps=tsteps,
                         n_frames=n_frames,
                         grayscale=grayscale)
sp.ttv_split(seed=seed)

n_cycles = 50
n_epochs = 20
visual_loss_fn = torch.nn.BCELoss()

vision_enc = FFVisualModuleEncoder(visual_embedding_dim=visual_embedding_dim, img_dim=img_dim, grayscale=grayscale, device=device)
vision_dec = FFVisualModuleDecoder(input_dim=visual_embedding_dim, threshold_dims=vision_enc.threshold_dims, grayscale=grayscale, device=device)

if train_from_previous:
    vision_enc.load_state_dict(torch.load(f'./trained_models/{recording_name}_vision_enc_{visual_embedding_dim}_{train_from_previous}.pth'))
    vision_dec.load_state_dict(torch.load(f'./trained_models/{recording_name}_vision_dec_{visual_embedding_dim}_{train_from_previous}.pth'))

def save_model():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(vision_enc.state_dict(), f'./trained_models/{recording_name}_vision_enc_{visual_embedding_dim}_{timestamp}.pth')
    torch.save(vision_dec.state_dict(), f'./trained_models/{recording_name}_vision_dec_{visual_embedding_dim}_{timestamp}.pth')
    logger.info(f'Saving trained vision modules timestamp: {timestamp}')


run = wandb.init(project="sensory-integration-visual-modules")
for cycle in tqdm(range(n_cycles)):
    vision_params = [vision_enc.parameters(), vision_dec.parameters()]
    optim = torch.optim.Adam(itertools.chain(*vision_params), lr=8*1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', 
                                                        factor=0.8, 
                                                        patience=100, 
                                                        threshold=0.01)
    
    # train vision modules
    vision_enc.train()
    vision_dec.train()
    for epoch in range(n_epochs):
        
        epoch_loss = 0
        
        for batch in range(len(sp.train_idxs) // batch_size):
            img, _, _ = sp.get_batches(batch_size=batch_size, ttv='tr')
            
            img = torch.as_tensor(img, dtype=torch.float32).view(-1, c_channels, 160, 120).to(device)
            img_recon = vision_dec(vision_enc(img))

            loss = visual_loss_fn(img_recon, img)

            optim.zero_grad()
            loss.backward()

            optim.step()
            epoch_loss += loss.item()
            scheduler.step(loss)
        
        print(f'epoch {epoch+1} img loss {np.mean(epoch_loss)}')
        wandb.log({'train_epoch_loss': np.mean(epoch_loss)})
        if scheduler.get_last_lr()[0] < 1e-9:
            break
        
    
    # evaluate on test set
    vision_enc.eval()
    vision_dec.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in range(len(sp.test_idxs) // batch_size):

            img, _, _ = sp.get_batches(batch_size=batch_size, ttv='te')
            img = torch.as_tensor(img, dtype=torch.float32).view(-1, c_channels, 160, 120).to(device)
            img_recon = vision_dec(vision_enc(img))
            loss = visual_loss_fn(img_recon, img)
            test_loss += loss.item()

        wandb.log({'test image batch': wandb.Image(np.swapaxes(img_recon[0].cpu().numpy(),0,2))})
        wandb.log({'test_epoch_loss': np.mean(test_loss)})
        print(f'cycle {cycle+1} test img loss {np.mean(test_loss)}')

    if os.path.exists('./stop_visual_training.txt'):
        save_model()
        break

save_model()



