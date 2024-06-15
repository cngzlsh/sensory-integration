from sensory_preprocessor import SensoryPreprocessor
import matplotlib.pyplot as plt
import librosa
import pathlib
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib

sns.set(font_scale=1.5)

recording_name = 'Movie_019'
has_trajectory = True
start_idx = 'random'

path_to_audio = pathlib.Path(f'./data/processed/{recording_name}/audio/audio.mp3')
path_to_video = pathlib.Path(f'./data/processed/{recording_name}/frames')
path_to_aux = pathlib.Path('./data/trajectories/300s_dt_0.02_switch_rect_riab_trajectory.json')
audio, sr = librosa.load(path_to_audio,sr=None, mono=False)
frames = [i for i in path_to_video.glob('*.jpg')]
tsteps = 10
grayscale = False

sp = SensoryPreprocessor(audio_path=path_to_audio,
                         video_path=path_to_video,
                         aux_path=path_to_aux,
                         tsteps=tsteps,
                         n_frames=15000,
                         grayscale=grayscale)

if start_idx == 'random':
    start_idx = np.random.randint(0, len(sp.aux['pos']))

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,10))
l_high = np.array([-np.inf for _ in range(442)])
r_high = np.array([-np.inf for _ in range(442)])

def update(frame):

    idx = frame + start_idx

    audio = sp.preprocess_audio(idx=idx)
    img = sp.preprocess_frame(idx=idx)
    aux = sp.preprocess_aux(idx=idx)
    pos = np.array(list(sp.aux['pos'][idx].values()))
        
    global l_high, r_high
    
    l_high = np.maximum(audio[:len(sp.freqs)], l_high)
    r_high = np.maximum(audio[len(sp.freqs):], r_high)

    ax[0,0].clear()
    ax[0,0].scatter(sp.freqs, audio[:len(sp.freqs)], s=15)
    # ax[0,0].scatter(sp.freqs, l_high, color='red', s=10, alpha=0.3)
    ax[0,0].set_title('L channel')
    ax[0,0].set_ylabel('dB')
    ax[0,0].set_ylim([-120, 30])

    ax[0,1].clear()
    ax[0,1].imshow(np.swapaxes(img,0,2))
    ax[0,1].set_title('Vision')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])

    ax[0,2].clear()
    ax[0,2].scatter(sp.freqs, audio[len(sp.freqs):], s=15)
    # ax[0,2].scatter(sp.freqs, r_high, color='red', s=10, alpha=0.3)
    ax[0,2].set_title('R channel')
    # ax[0,2].set_xlabel('Freqs')
    ax[0,2].set_ylabel('dB')
    ax[0,2].set_ylim([-120, 30])

    ax[1,0].clear()
    ax[1,0].scatter(x=[[0.1]], y=[[0.1]], marker='x', s=150, label='Low')
    ax[1,0].scatter(x=[[0.9]], y=[[1.9]], marker='x', s=150, label='High')
    ax[1,0].set_title('Position in arena')
    ax[1,0].legend()
    ax[1,0].set_xlim([0,1])
    ax[1,0].set_ylim([0,2])
    if has_trajectory:
        ax[1,0].scatter(pos[0,0], pos[0,1], s=150)

    ax[1,1].clear()
    ax[1,1].set_title(f'Frame: {idx}')
    ax[1,1].set_xlim([-1,1])
    ax[1,1].set_ylim([-1,1])
    ax[1,1].set_xticks([])
    ax[1,1].set_yticks([])
    if has_trajectory:
        ax[1,1].plot([0, aux[3]], [0, aux[4]], linewidth=10)
        

    ax[1,2].clear()
    ax[1,2].set_title('Velocity')
    if has_trajectory:
        ax[1,2].bar(['X','Y'],[aux[0], aux[1]])
        ax[1,2].set_ylim([sp.minvel, sp.maxvel])
        ax[1,2].set_xticks(['X', 'Y'])
    return ax


animation = FuncAnimation(fig, update, frames=1000, interval=25)
plt.show()


writervideo = matplotlib.animation.FFMpegWriter(fps=24)
animation.save('./Movie_019.mp4', writer=writervideo)