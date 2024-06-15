import numpy as np
import librosa
from PIL import Image

def load_stereo_audio(audio_path):
    """
    Load a stereo audio file and return the two channels separately.
    """
    # Load the audio file. librosa automatically resamples to 22050 Hz by default.
    # Setting `mono=False` keeps the stereo channels.
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # Ensure the audio array is in the shape (2, n_samples) for stereo
    if y.ndim == 1:  # This means audio is mono despite setting mono=False
        y = np.tile(y, (2, 1))  # Duplicate the mono track to simulate stereo
    elif y.shape[0] != 2:  # Make sure it's (2, n_samples)
        y = y.T
    
    return y, sr

def process_stereo_audio(y, sr, n_fft=2048):
    """
    Perform simple audio processing on stereo audio.
    Here, we compute the Short-Time Fourier Transform (STFT) of each channel.
    """
    # Compute STFT for each channel
    D_left = librosa.stft(y[0, :], n_fft=n_fft)  # Left channel
    D_right = librosa.stft(y[1, :], n_fft=n_fft)  # Right channel
    
    # Example processing: Compute the magnitude spectrograms
    S_left = np.abs(D_left)
    S_right = np.abs(D_right)
    
    return S_left, S_right

def compute_ild_itd(audio, sr=44100):
    '''
    Returns the mean interaural level difference (ILD) and interaural time difference (ITD) of a short audio segment.
    Calculate ILD by comparing RMS levels of the two channels.
    Calculate ITD by finding the lag that maximizes the cross-correlation.
    Audio must be 2-channel stereo.
    '''
    if audio.shape[0] != 2:
        raise ValueError("Audio file is not stereo")
    
    # Split the channels
    left_channel = audio[0, :]
    right_channel = audio[1, :]

    # ILD
    ild = 20 * np.log10(np.sqrt(np.mean(np.square(left_channel))) / np.sqrt(np.mean(np.square(right_channel))))

    # ITD
    correlation = np.correlate(left_channel, right_channel, "full")
    lag = np.argmax(correlation) - (len(right_channel) - 1)
    itd = lag / float(sr)  # Convert sample lag into time

    return ild, itd

def compute_frequency_representation(audio, sr=44100, channel='mono'):
    '''
    Performs FFT on the audio segment.
    Returns the magnitude in dB in frequency domain. Dimension (2, 1+N/2) if stereo or (1+N/2) if mono
    where N is the length of audio
    '''
    if channel == 'mono':
        if len(audio.shape) == 2: # if provided stereo audio, take the main of the first axis
            audio = audio.mean(axis=0)

        fft_spectrum = np.fft.rfft(audio)
        frequencies = np.fft.rfftfreq(len(audio), 1/sr)

        magnitude_spectrum = np.abs(fft_spectrum)
        magnitude_dB = 20 * np.log10(magnitude_spectrum + np.finfo(float).eps)  # add eps for numerical stability

        return frequencies, magnitude_dB
    elif channel == 'stereo':
        assert audio.shape[0] == 2
        assert len(audio.shape) == 2

        audio_l = audio[0,:]
        audio_r = audio[1,:]
        fft_spectrum_l, fft_spectrum_r = np.fft.rfft(audio_l), np.fft.rfft(audio_r)
        
        frequencies= np.fft.rfftfreq(len(audio_l), 1/sr)

        magnitude_spectrum_l, magnitude_spectrum_r = np.abs(fft_spectrum_l), np.abs(fft_spectrum_r)
        magnitude_dB_l, magnitude_dB_r = 20 * np.log10(magnitude_spectrum_l + np.finfo(float).eps), 20 * np.log10(magnitude_spectrum_r + np.finfo(float).eps)  # add eps for numerical stability

        return frequencies, (magnitude_dB_l, magnitude_dB_r)
        
    else:
        raise ValueError('specify channel as mono or stereo')