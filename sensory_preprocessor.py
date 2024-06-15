import librosa
import numpy as np
import json
from PIL import Image
from pathlib import Path
from utils import compute_frequency_representation

class SensoryPreprocessor:
    '''
    Sensory Preprocessor batch: preprocesses frames and audios and then samples batches of (frames, audio, auxiliary variables)
    '''
    def __init__(self, audio_path, video_path, aux_path, n_frames=None, tsteps=10, grayscale=False, has_trajectory=True):
        self.audio_path = audio_path
        self.video_path = video_path
        self.aux_path = aux_path
        self.tsteps = tsteps
        self.grayscale = grayscale
        self.has_trajectory = has_trajectory

        self.audio, self.sr = librosa.load(self.audio_path, sr=None, mono=False)
        self.frames_glob = [i for i in self.video_path.glob('*.jpg')]               

        self.n_frames = len(self.frames_glob) if n_frames is None else n_frames     # actual recorded frames vs frames simulated with aux
        if self.n_frames == 0:
            assert FileNotFoundError('check video frames path')
        self.frame_to_audio_ratio = self.audio.shape[1] / len(self.frames_glob)
        self.preprocess_audio(idx=1) # to get the freqs

        if self.has_trajectory:
            self.aux = json.loads(open(aux_path).read())
            self.maxvel, self.minvel = np.max(np.vstack([list(x.values()) for x in self.aux['vel']])), np.min(np.vstack([list(x.values()) for x in self.aux['vel']]))
            self.maxrotvel, self.minrotvel = np.max(self.aux['rot_vel']), np.min(self.aux['rot_vel'])
        self.pointer = 0

    def get_img_dim(self):
        # queries the first frame and returns its dimensions
        return np.array(Image.open(self.frames_glob[0])).shape[:2]
    
    def ttv_split(self, ratio=(0.8, 0.1, 0.1), seed=1234):
        from sklearn.model_selection import train_test_split # type: ignore
        train_valid_idxs, self.test_idxs = train_test_split(np.arange(self.n_frames // self.tsteps), test_size=ratio[1], random_state=seed)
        if len(ratio) == 2:
            self.train_idxs = train_valid_idxs
        else:
            self.train_idxs, self.valid_idxs = train_test_split(train_valid_idxs, test_size=ratio[2]/(ratio[0] + ratio[2]), random_state=seed)
        self.tr_idx, self.te_idx, self.va_idx = 0, 0, 0

        assert np.intersect1d(self.train_idxs, self.test_idxs).size == 0, 'train and test sets intersect'
        assert np.intersect1d(self.train_idxs, self.valid_idxs).size == 0, 'train and valid sets intersect'
        assert np.intersect1d(self.valid_idxs, self.test_idxs).size == 0, 'valid and test sets intersect'

    def preprocess_audio(self, idx=None):
        # process audio: performs FFT and returns freqency power for each channel respectively
        if idx == None:
            idx = self.pointer

        self.freqs, (magnitude_dB_l, magnitude_dB_r) = compute_frequency_representation(
            self.audio[:, int(idx * self.frame_to_audio_ratio): int((idx+1) * self.frame_to_audio_ratio)],
            sr=self.sr,
            channel='stereo')
        
        return np.concatenate([magnitude_dB_l, magnitude_dB_r])

    def preprocess_frame(self, idx=None):
        # process frame: transforms into grayscale if required, normalise image
        if idx == None:
            idx = self.pointer
        
        img = Image.open(self.frames_glob[idx])
        if self.grayscale:
            img = img.convert('L')
        img = np.array(img) / 255.0
        if self.grayscale:
            img = img[..., None]
        
        img = np.swapaxes(img, 0, 2) # (3, w, l) if RGB, (1, w, l) if grayscale
        return  img 
    
    def preprocess_aux(self, idx=None):
        # concatenates velocity, rotational velocity and head direction
        if not self.has_trajectory:
            return np.zeros(5) # returns an array of zeros if no trajectory is provided
        
        if idx == None:
            idx = self.pointer
        vel = np.array(list(self.aux['vel'][idx].values()))
        rot_vel = np.array([self.aux['rot_vel'][idx]])
        hd = np.array(list(self.aux['head_direction'][idx].values()))

        return np.concatenate([vel[0], rot_vel, hd[0]])

    def get_batches(self, batch_size=1, random=False, ttv=None, get_pos=False):
        '''
        Samples a single batch of (batch size) video frames, audios and auxs
        params:
            batch_size:     size of batch
            random:         whether to randomise samples
            ttv_split:      whether to sample from train set, test set or validation set.
            get_pos:        whether to also return the normalised position of the agent.
        '''
        if batch_size > self.n_frames // self.tsteps:
            raise ValueError(f'batch size is bigger than total number of batches ({self.n_frames // self.tsteps})')
        # returns: (bs, imgs) (bs, audios) (bs, auxs)
        # if random is set to True, samples randomly, otherwise sample in order by pointer location
        imgs, audios, auxs, pos = [], [], [], []

        for _ in range(batch_size):
            # whether to sample from specific set: if so extract the current pointer. if both random and ttv are False sample sequentially
            if ttv is None or ttv is False:
                idx = self.tsteps * (np.random.randint(0, self.n_frames-self.tsteps) // self.tsteps) if random else self.pointer
            else:
                if ttv[:2] == 'tr':
                    idx = self.train_idxs[self.tr_idx] * self.tsteps
                elif ttv[:2] == 'te':
                    idx = self.test_idxs[self.te_idx] * self.tsteps
                elif ttv[:2] == 'va':
                    idx = self.test_idxs[self.va_idx] * self.tsteps
                else:
                    raise ValueError
            
            batch_imgs, batch_audios, batch_auxs, batch_pos = [], [], [], []
            for t in range(self.tsteps):
                batch_imgs.append(self.preprocess_frame(idx+t))
                batch_audios.append(self.preprocess_audio(idx+t))
                batch_auxs.append(self.preprocess_aux(idx+t))
                batch_pos.append(np.array(list(self.aux['pos'][idx+t].values()))[None,:])
            imgs.append(np.stack(batch_imgs, axis=0))
            audios.append(np.stack(batch_audios, axis=0))
            auxs.append(np.stack(batch_auxs, axis=0))
            pos.append(np.stack(batch_pos, axis=0))

            if ttv is None or ttv is False:
                if not random: 
                        self.pointer += self.tsteps
                        self.pointer %= self.n_frames - self.tsteps
            else:
                if ttv[:2] == 'tr':
                    self.tr_idx += 1
                    self.tr_idx %= len(self.train_idxs)
                elif ttv[:2] == 'te':
                    self.te_idx += 1
                    self.te_idx %= len(self.test_idxs)
                elif ttv[:3] == 'va':
                    self.va_idx += 1
                    self.va_idx %= len(self.valid_idxs)
        if get_pos:
            return np.stack(imgs, axis=0), np.stack(audios, axis=0), np.stack(auxs, axis=0), np.stack(pos, axis=0)
        else:
            return np.stack(imgs, axis=0), np.stack(audios, axis=0), np.stack(auxs, axis=0)
    
    def reset_pointers(self):
        self.pointer, self.tr_idx, self.te_idx, self.va_idx = 0, 0, 0, 0
