#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
# Authors: Shengkui Zhao, Zexu Pan

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import numpy as np
import torchaudio
from .misc import stft, istft, compute_fbank

# Constant for normalizing audio values
MAX_WAV_VALUE = 32768.0

def decode_one_audio(model, device, inputs, args):
    """Decodes audio using the specified model based on the provided network type.

    This function selects the appropriate decoding function based on the specified
    network in the arguments and processes the input audio data accordingly.

    Args:
        model (nn.Module): The trained model used for decoding.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        inputs (torch.Tensor): Input audio tensor.
        args (Namespace): Contains arguments for network configuration.

    Returns:
        list: A list of decoded audio outputs for each speaker.
    """
    # Select decoding function based on the network type specified in args
    if args.network == 'MossFormer2_SE_48K':
        return decode_one_audio_mossformer2_se_48k(model, device, inputs, args)
    else:
        print("No network found!")  # Print error message if no valid network is specified
        return 

def decode_one_audio_mossformer2_se_48k(model, device, inputs, args):
    """Processes audio inputs through the MossFormer2 model for speech enhancement at 48kHz.

    This function decodes audio input using the following steps:
    1. Normalizes the audio input to a maximum WAV value.
    2. Checks the length of the input to decide between online decoding and batch processing.
    3. For longer inputs, processes the audio in segments using a sliding window.
    4. Computes filter banks and their deltas for the audio segment.
    5. Passes the filter banks through the model to get a predicted mask.
    6. Applies the mask to the spectrogram of the audio segment and reconstructs the audio.
    7. For shorter inputs, processes them in one go without segmentation.
    
    Args:
        model (nn.Module): The trained MossFormer2 model used for decoding.
        device (torch.device): The device (CPU or GPU) for computation.
        inputs (torch.Tensor): Input audio tensor of shape (B, T), where B is the batch size and T is the number of time steps.
        args (Namespace): Contains arguments for sampling rate, window size, and other parameters.

    Returns:
        numpy.ndarray: The decoded audio output, normalized to the range [-1, 1].
    """
    inputs = inputs[0, :]  # Extract the first element from the input tensor
    input_len = inputs.shape[0]  # Get the length of the input audio
    inputs = inputs * MAX_WAV_VALUE  # Normalize the input to the maximum WAV value

    # Check if input length exceeds the defined threshold for online decoding
    if input_len > args.sampling_rate * args.one_time_decode_length:  # 20 seconds
        online_decoding = True
        if online_decoding:
            window = int(args.sampling_rate * args.decode_window)  # Define window length (e.g., 4s for 48kHz)
            stride = int(window * 0.75)  # Define stride length (e.g., 3s for 48kHz)
            t = inputs.shape[0]  # Update length after potential padding

            # Pad input if necessary to match window size
            if t < window:
                inputs = np.concatenate([inputs, np.zeros(window - t)], 0)
            elif t < window + stride:
                padding = window + stride - t
                inputs = np.concatenate([inputs, np.zeros(padding)], 0)
            else:
                if (t - window) % stride != 0:
                    padding = t - (t - window) // stride * stride
                    inputs = np.concatenate([inputs, np.zeros(padding)], 0)

            audio = torch.from_numpy(inputs).type(torch.FloatTensor)  # Convert to Torch tensor
            t = audio.shape[0]  # Update length after conversion
            outputs = torch.from_numpy(np.zeros(t))  # Initialize output tensor
            give_up_length = (window - stride) // 2  # Determine length to ignore at the edges
            dfsmn_memory_length = 0  # Placeholder for potential memory length
            current_idx = 0  # Initialize current index for sliding window

            # Process audio in sliding window segments
            while current_idx + window <= t:
                # Select appropriate segment of audio for processing
                if current_idx < dfsmn_memory_length:
                    audio_segment = audio[0:current_idx + window]
                else:
                    audio_segment = audio[current_idx - dfsmn_memory_length:current_idx + window]

                # Compute filter banks for the audio segment
                fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
                
                # Compute deltas for filter banks
                fbank_tr = torch.transpose(fbanks, 0, 1)  # Transpose for delta computation
                fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)  # First-order delta
                fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)  # Second-order delta
                
                # Transpose back to original shape
                fbank_delta = torch.transpose(fbank_delta, 0, 1)
                fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

                # Concatenate the original filter banks with their deltas
                fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
                fbanks = fbanks.unsqueeze(0).to(device)  # Add batch dimension and move to device

                # Pass filter banks through the model
                Out_List = model(fbanks)
                pred_mask = Out_List[-1]  # Get the predicted mask from the output

                # Apply STFT to the audio segment
                spectrum = stft(audio_segment, args)
                pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
                masked_spec = spectrum.cpu() * pred_mask.detach().cpu()  # Apply mask to the spectrum
                masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form

                # Reconstruct audio from the masked spectrogram
                output_segment = istft(masked_spec_complex, args, len(audio_segment))

                # Store the output segment in the output tensor
                if current_idx == 0:
                    outputs[current_idx:current_idx + window - give_up_length] = output_segment[:-give_up_length]
                else:
                    output_segment = output_segment[-window:]  # Get the latest window of output
                    outputs[current_idx + give_up_length:current_idx + window - give_up_length] = output_segment[give_up_length:-give_up_length]
                
                current_idx += stride  # Move to the next segment

    else:
        # Process the entire audio at once if it is shorter than the threshold
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)
        fbanks = compute_fbank(audio.unsqueeze(0), args)

        # Compute deltas for filter banks
        fbank_tr = torch.transpose(fbanks, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)

        # Concatenate the original filter banks with their deltas
        fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)
        fbanks = fbanks.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Pass filter banks through the model
        Out_List = model(fbanks)
        pred_mask = Out_List[-1]  # Get the predicted mask
        spectrum = stft(audio, args)  # Apply STFT to the audio
        pred_mask = pred_mask.permute(2, 1, 0)  # Permute dimensions for masking
        masked_spec = spectrum * pred_mask.detach().cpu()  # Apply mask to the spectrum
        masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]  # Convert to complex form
        
        # Reconstruct audio from the masked spectrogram
        outputs = istft(masked_spec_complex, args, len(audio))

    return outputs.numpy() / MAX_WAV_VALUE  # Return the output normalized to [-1, 1]
