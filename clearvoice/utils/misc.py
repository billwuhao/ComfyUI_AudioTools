#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Import future compatibility features for Python 2/3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import necessary libraries
import torch 
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed
import os 
import sys
import librosa  # Library for audio processing
import torchaudio  # Library for audio processing with PyTorch

# Constants
MAX_WAV_VALUE = 32768.0  # Maximum value for WAV files
EPS = 1e-6  # Small value to avoid division by zero

def read_and_config_file(input_path, decode=0):
    """Reads input paths from a file or directory and configures them for processing.

    Args:
        input_path (str): Path to the input directory or file.
        decode (int): Flag indicating if decoding should occur (1 for decode, 0 for standard read).

    Returns:
        list: A list of processed paths or dictionaries containing input and label paths.
    """
    processed_list = []

    # If decoding is requested, find files in a directory
    if decode:
        if os.path.isdir(input_path):
            processed_list = librosa.util.find_files(input_path, ext="wav")  # Look for WAV files
            if len(processed_list) == 0:
                processed_list = librosa.util.find_files(input_path, ext="flac")  # Fallback to FLAC files
        else:
            # Read paths from a file
            with open(input_path) as fid:
                for line in fid:
                    path_s = line.strip().split()  # Split line into parts
                    processed_list.append(path_s[0])  # Append the first part (input path)
        return processed_list

    # Read input-label pairs from a file
    with open(input_path) as fid:
        for line in fid:
            tmp_paths = line.strip().split()  # Split line into parts
            if len(tmp_paths) == 3:  # Expecting input, label, and duration
                sample = {'inputs': tmp_paths[0], 'labels': tmp_paths[1], 'duration': float(tmp_paths[2])}
            elif len(tmp_paths) == 2:  # Expecting input and label only
                sample = {'inputs': tmp_paths[0], 'labels': tmp_paths[1]}
            processed_list.append(sample)  # Append the sample dictionary
    return processed_list

def load_checkpoint(checkpoint_path, use_cuda):
    """Loads the model checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        use_cuda (bool): Flag indicating whether to use CUDA for loading.

    Returns:
        dict: The loaded checkpoint containing model parameters.
    """
    #if use_cuda:
    #    checkpoint = torch.load(checkpoint_path)  # Load using CUDA
    #else:
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)  # Load to CPU
    return checkpoint

def get_learning_rate(optimizer):
    """Retrieves the current learning rate from the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer instance.

    Returns:
        float: The current learning rate.
    """
    return optimizer.param_groups[0]["lr"]

def reload_for_eval(model, checkpoint_dir, use_cuda):
    """Reloads a model for evaluation from the specified checkpoint directory.

    Args:
        model (nn.Module): The model to be reloaded.
        checkpoint_dir (str): Directory containing checkpoints.
        use_cuda (bool): Flag indicating whether to use CUDA.

    Returns:
        None
    """
    print('Reloading from: {}'.format(checkpoint_dir))
    best_name = os.path.join(checkpoint_dir, 'last_best_checkpoint')  # Path to the best checkpoint
    ckpt_name = os.path.join(checkpoint_dir, 'last_checkpoint')  # Path to the last checkpoint
    if os.path.isfile(best_name):
        name = best_name 
    elif os.path.isfile(ckpt_name):
        name = ckpt_name
    else:
        print('Warning: No existing checkpoint or best_model found!')
        return
    
    with open(name, 'r') as f:
        model_name = f.readline().strip()  # Read the model name from the checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, model_name)  # Construct full checkpoint path
    print('Checkpoint path: {}'.format(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path, use_cuda)  # Load the checkpoint
    '''
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)  # Load model parameters
    else:
        model.load_state_dict(checkpoint, strict=False)
    '''
    if 'model' in checkpoint:
        pretrained_model = checkpoint['model']
    else:
        pretrained_model = checkpoint
    state = model.state_dict()
    for key in state.keys():
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            state[key] = pretrained_model[key]
        elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:
             state[key] = pretrained_model[key.replace('module.', '')]
        elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
             state[key] = pretrained_model['module.'+key]
        elif self.print: print(f'{key} not loaded')
    model.load_state_dict(state)

    print('=> Reload well-trained model {} for decoding.'.format(model_name))
    

def reload_model(model, optimizer, checkpoint_dir, use_cuda=True, strict=True):
    """Reloads the model and optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model to be reloaded.
        optimizer (torch.optim.Optimizer): The optimizer to be reloaded.
        checkpoint_dir (str): Directory containing checkpoints.
        use_cuda (bool): Flag indicating whether to use CUDA.
        strict (bool): If True, requires keys in state_dict to match exactly.

    Returns:
        tuple: Current epoch and step.
    """
    ckpt_name = os.path.join(checkpoint_dir, 'checkpoint')  # Path to the checkpoint file
    if os.path.isfile(ckpt_name):
        with open(ckpt_name, 'r') as f:
            model_name = f.readline().strip()  # Read model name from checkpoint file
        checkpoint_path = os.path.join(checkpoint_dir, model_name)  # Construct full checkpoint path
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)  # Load the checkpoint
        model.load_state_dict(checkpoint['model'], strict=strict)  # Load model parameters
        optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer parameters
        epoch = checkpoint['epoch']  # Get current epoch
        step = checkpoint['step']  # Get current step
        print('=> Reloaded previous model and optimizer.')
    else:
        print('[!] Checkpoint directory is empty. Train a new model ...')
        epoch = 0  # Initialize epoch
        step = 0  # Initialize step
    return epoch, step

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, mode='checkpoint'):
    """Saves the model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        epoch (int): Current epoch number.
        step (int): Current training step number.
        checkpoint_dir (str): Directory to save the checkpoint.
        mode (str): Mode of the checkpoint ('checkpoint' or other).

    Returns:
        None
    """
    checkpoint_path = os.path.join(
        checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(epoch, step))  # Construct checkpoint file path
    torch.save({'model': model.state_dict(),  # Save model parameters
                'optimizer': optimizer.state_dict(),  # Save optimizer parameters
                'epoch': epoch,  # Save epoch
                'step': step}, checkpoint_path)  # Save checkpoint to file

    # Save the checkpoint name to a file for easy access
    with open(os.path.join(checkpoint_dir, mode), 'w') as f:
        f.write('model.ckpt-{}-{}.pt'.format(epoch, step))
    print("=> Saved checkpoint:", checkpoint_path)

def setup_lr(opt, lr):
    """Sets the learning rate for all parameter groups in the optimizer.

    Args:
        opt (torch.optim.Optimizer): The optimizer instance whose learning rate needs to be set.
        lr (float): The new learning rate to be assigned.
    
    Returns:
        None
    """
    for param_group in opt.param_groups:
        param_group['lr'] = lr  # Update the learning rate for each parameter group

def power_compress(x):
    """Compresses the power of a complex spectrogram.

    Args:
        x (torch.Tensor): Input tensor with real and imaginary components.

    Returns:
        torch.Tensor: Compressed magnitude and phase representation of the input.
    """
    real = x[..., 0]  # Extract real part
    imag = x[..., 1]  # Extract imaginary part
    spec = torch.complex(real, imag)  # Create complex tensor from real and imaginary parts
    mag = torch.abs(spec)  # Compute magnitude
    phase = torch.angle(spec)  # Compute phase
    
    mag = mag**0.3  # Compress magnitude using power of 0.3
    real_compress = mag * torch.cos(phase)  # Reconstruct real part
    imag_compress = mag * torch.sin(phase)  # Reconstruct imaginary part
    return torch.stack([real_compress, imag_compress], 1)  # Stack compressed parts


def power_uncompress(real, imag):
    """Uncompresses the power of a compressed complex spectrogram.

    Args:
        real (torch.Tensor): Compressed real component.
        imag (torch.Tensor): Compressed imaginary component.

    Returns:
        torch.Tensor: Uncompressed complex spectrogram.
    """
    spec = torch.complex(real, imag)  # Create complex tensor from real and imaginary parts
    mag = torch.abs(spec)  # Compute magnitude
    phase = torch.angle(spec)  # Compute phase
    
    mag = mag**(1./0.3)  # Uncompress magnitude by raising to the power of 1/0.3
    real_uncompress = mag * torch.cos(phase)  # Reconstruct real part
    imag_uncompress = mag * torch.sin(phase)  # Reconstruct imaginary part
    return torch.stack([real_uncompress, imag_uncompress], -1)  # Stack uncompressed parts


def stft(x, args, center=False, periodic=False, onesided=None):
    """Computes the Short-Time Fourier Transform (STFT) of an audio signal.

    Args:
        x (torch.Tensor): Input audio signal.
        args (Namespace): Configuration arguments containing window type and lengths.
        center (bool): Whether to center the window.

    Returns:
        torch.Tensor: The computed STFT of the input signal.
    """
    win_type = args.win_type
    win_len = args.win_len
    win_inc = args.win_inc
    fft_len = args.fft_len

    # Select window type and create window tensor
    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=periodic).to(x.device)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=periodic).to(x.device)
    else:
        print(f"In STFT, {win_type} is not supported!")
        return

    # Compute and return the STFT
    return torch.stft(x, fft_len, win_inc, win_len, center=center, window=window, onesided=onesided, return_complex=False)

def istft(x, args, slen=None, center=False, normalized=False, periodic=False, onesided=None, return_complex=False):
    """Computes the inverse Short-Time Fourier Transform (ISTFT) of a complex spectrogram.

    Args:
        x (torch.Tensor): Input complex spectrogram.
        args (Namespace): Configuration arguments containing window type and lengths.
        slen (int, optional): Length of the output signal.
        center (bool): Whether to center the window.
        normalized (bool): Whether to normalize the output.
        onesided (bool, optional): If True, computes only the one-sided transform.
        return_complex (bool): If True, returns complex output.

    Returns:
        torch.Tensor: The reconstructed audio signal from the spectrogram.
    """
    win_type = args.win_type
    win_len = args.win_len
    win_inc = args.win_inc
    fft_len = args.fft_len

    # Select window type and create window tensor
    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=periodic).to(x.device)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=periodic).to(x.device)
    else:
        print(f"In ISTFT, {win_type} is not supported!")
        return

    try:
        # Attempt to compute ISTFT
        output = torch.istft(x, n_fft=fft_len, hop_length=win_inc, win_length=win_len,
                              window=window, center=center, normalized=normalized,
                              onesided=onesided, length=slen, return_complex=False)
    except:
        # Handle potential errors by converting x to a complex tensor
        x_complex = torch.view_as_complex(x)
        output = torch.istft(x_complex, n_fft=fft_len, hop_length=win_inc, win_length=win_len,
                              window=window, center=center, normalized=normalized,
                              onesided=onesided, length=slen, return_complex=False)
    return output

def compute_fbank(audio_in, args):
    """Computes the filter bank features from an audio signal.

    Args:
        audio_in (torch.Tensor): Input audio signal.
        args (Namespace): Configuration arguments containing window length, shift, and sampling rate.

    Returns:
        torch.Tensor: Computed filter bank features.
    """
    frame_length = args.win_len / args.sampling_rate * 1000  # Frame length in milliseconds
    frame_shift = args.win_inc / args.sampling_rate * 1000  # Frame shift in milliseconds

    # Compute and return filter bank features using Kaldi's implementation
    return torchaudio.compliance.kaldi.fbank(audio_in, dither=1.0, frame_length=frame_length,
                                             frame_shift=frame_shift, num_mel_bins=args.num_mels,
                                             sample_frequency=args.sampling_rate, window_type=args.win_type)
                                             


