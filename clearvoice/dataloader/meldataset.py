import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
import scipy
import librosa
import wave
from pydub import AudioSegment

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    try:
        sampling_rate, data = read(full_path)
        if max(data.shape) / sampling_rate < 0.5:
            return None, None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

    if len(data.shape) > 1:
        if data.shape[1] <= 2:
            data = data[...,0]
        else:
            data = data[0,...]
    return data / MAX_WAV_VALUE, sampling_rate

def get_wave_duration(file_path):
    """
    Gets the duration of a WAV file in seconds.

    :param file_path: Path to the WAV file.
    :return: Duration of the WAV file in seconds.
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            # Get the number of frames
            num_frames = wf.getnframes()
            # Get the frame rate
            frame_rate = wf.getframerate()
            # Calculate duration
            duration = num_frames / float(frame_rate)
            return duration, frame_rate, num_frames
    except wave.Error as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

def read_audio_segment(file_path, start_ms, end_ms):
    """
    Reads a segment from a WAV file and returns the raw data and its properties.

    :param file_path: Path to the WAV file.
    :param start_ms: Start time of the segment in milliseconds.
    :param end_ms: End time of the segment in milliseconds.
    :return: A tuple containing the raw audio data, frame rate, sample width, and number of channels.
    """
    #start_time = time.time()
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(file_path)
        # Extract the segment
        segment = audio[start_ms:end_ms]
        # Get raw audio data
        raw_data = segment.raw_data
        # Get audio properties
        frame_rate = segment.frame_rate
        sample_width = segment.sample_width
        channels = segment.channels
        # Create NumPy array from the raw audio data
        audio_array = np.frombuffer(raw_data, dtype=np.int16)

        # If stereo, reshape the array to have a second dimension
        if channels > 1:
            audio_array = audio_array.reshape((-1, channels))
            audio_array = audio_array[...,0]
        '''
        if frame_rate !=48000:
            audio_array = audio_array/MAX_WAV_VALUE
            audio_array = librosa.resample(audio_array, frame_rate, 48000)
            audio_array = audio_array * MAX_WAV_VALUE
            frame_rate = 48000
        '''
        #end_time = time.time()
        #time_taken = end_time - start_time

        #print(f"Successfully read segment from {start_ms}ms to {end_ms}ms in {time_taken:.4f} seconds")
        return audio_array / MAX_WAV_VALUE#, frame_rate #, sample_width, channels
    except Exception as e:
        print(f"An error occurred: {e}")
        return None#, None #, None, None

def resample(audio, sr_in, sr_out, target_len=None):
    #audio = audio / MAX_WAV_VALUE
    #audio = normalize(audio) * 0.95
    if target_len is not None:
        audio = scipy.signal.resample(audio, target_len)
        return audio
    resample_factor = sr_out / sr_in
    new_samples = int(len(audio) * resample_factor)
    audio = scipy.signal.resample(audio, new_samples)
    return audio

def load_segment(full_path, target_sampling_rate=None, segment_size=None):

    if segment_size is not None:
        dur,sampling_rate,len_data = get_wave_duration(full_path)
        if sampling_rate is None: return None, None
        if sampling_rate < 44100: return None, None

        target_dur = segment_size / target_sampling_rate
        if dur < target_dur:
            data, sampling_rate = load_wav(full_path)
            #print(f'data_read: {data.shape}, sampling_rate: {sampling_rate}')
            if data is None: return None, None

            if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
                data = resample(data, sampling_rate, target_sampling_rate)
                sampling_rate = target_sampling_rate
            data = torch.FloatTensor(data)
            data = data.unsqueeze(0)
            data = torch.nn.functional.pad(data, (0, segment_size - data.size(1)), 'constant')
            data = data.squeeze(0)
            return data.numpy(), sampling_rate
        else:
            dur,sampling_rate,len_data = get_wave_duration(full_path)
            if sampling_rate < 44100: return None, None
                        
            target_dur = segment_size / target_sampling_rate
            target_len = int(target_dur * sampling_rate)
            start_idx = random.randint(0, (len_data - target_len))
            start_ms = start_idx / sampling_rate * 1000
            end_ms = start_ms + target_dur * 1000
            data = read_audio_segment(full_path, start_ms, end_ms)    
            #print(f'data_read: {data.shape}, sampling_rate: {sampling_rate}')
            if data is None: return None, None
            if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
                data = resample(data, sampling_rate, target_sampling_rate)
                sampling_rate = target_sampling_rate
            if len(data) < segment_size:
                data = torch.FloatTensor(data)
                data = data.unsqueeze(0)
                data = torch.nn.functional.pad(data, (0, segment_size - data.size(1)), 'constant')
                data = data.squeeze(0)
                data = data.numpy()
            else:
                start_idx = random.randint(0, (len(data) - segment_size))
                data = data[start_idx:start_idx+segment_size]
            #print(f'data_cut: {data.shape}')
            return data, sampling_rate
    else:
        dur,sampling_rate,len_data = get_wave_duration(full_path)
        if sampling_rate is None: return None, None
        if sampling_rate < 44100: return None, None
        data, sampling_rate = load_wav(full_path)
        if data is None: return None, None
        if target_sampling_rate is not None and sampling_rate != target_sampling_rate:
            data = resample(data, sampling_rate, target_sampling_rate)
            sampling_rate = target_sampling_rate
        return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    '''
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    '''
    global mel_basis, hann_window
    if fmax not in mel_basis:
        #mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        # sr, n_fft, n_mels=128, fmin=0.0, fmax
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist_org(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]

    return training_files, validation_files

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.supported_samples = [16000, 22050, 24000] #[4000, 8000, 16000, 22050, 24000, 32000]
        #self.supported_samples = [4000, 8000] #, 16000, 22050, 24000, 32000]

    def __getitem__(self, index):
        filename = self.audio_files[index]
        while 1:
            #audio, sampling_rate = load_wav(filename)
            audio, sampling_rate = load_segment(filename, self.sampling_rate, self.segment_size)
            if audio is not None: break
            else:
                filename = self.audio_files[random.randint(0,index)]
                #audio, sampling_rate = load_wav(filename)
                #audio, sampling_rate = load_segment(filename, self.sampling_rate, self.segment_size)

        #audio = audio / MAX_WAV_VALUE
        if not self.fine_tuning:
            audio = normalize(audio) * 0.95
            
        sr_out = random.choice(self.supported_samples)
        audio_down = resample(audio, self.sampling_rate, sr_out)
            
        target_len = len(audio) #/ downsample_factor
        audio_up = resample(audio_down, None, None, target_len)

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        audio_up = torch.FloatTensor(audio_up)
        audio_up = audio_up.unsqueeze(0)

        mel = mel_spectrogram(audio_up, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __getitem__org(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            while 1:
                audio, sampling_rate = load_wav(filename)
                if audio is not None: break
                else:
                    filename = self.audio_files[random.randint(0,index)]
                    audio, sampling_rate = load_wav(filename)

            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            #self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                resample_factor = self.sampling_rate / sampling_rate
                new_samples = int(len(audio) * resample_factor)
                audio = scipy.signal.resample(audio, new_samples)#.astype(np.int16)
                #raise ValueError("{} SR doesn't match target {} SR".format(
                #    sampling_rate, self.sampling_rate))
            
            downsample_factor = 16000 / self.sampling_rate
            new_samples = int(len(audio) * downsample_factor)
            audio_down = scipy.signal.resample(audio, new_samples)
            
            new_samples = len(audio) #/ downsample_factor
            audio_up = scipy.signal.resample(audio_down, new_samples)
            #print(f'audio: {audio.shape}, audio_up: {audio_up.shape}') 
            #min_idx = min(len(audio), len(audio_up))
            #audio = audio[:min_idx]
            #audio_up = audio_up[:min_idx]

            self.cached_wav = audio
            self.cached_wav_up = audio_up
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            audio_up = self.cached_wav_up
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        audio_up = torch.FloatTensor(audio_up)
        audio_up = audio_up.unsqueeze(0)

        if True:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                    audio_up = audio_up[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
                    audio_up = torch.nn.functional.pad(audio_up, (0, self.segment_size - audio_up.size(1)), 'constant')

            mel = mel_spectrogram(audio_up, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
