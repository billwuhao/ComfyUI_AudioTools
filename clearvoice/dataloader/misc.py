
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import librosa
import mimetypes

def get_file_extension(file_path):
    """
    Return an audio file extension
    """

    _, ext = os.path.splitext(file_path)
    return ext

def is_audio_file(file_path):
    """
    Check if the given file_path is an audio file
    Return True if it is an audio file, otherwise, return False
    """
    file_ext = ["wav", "aac", "ac3", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wma", "webm"]

    ext = get_file_extension(file_path)
    if ext.replace('.','') in file_ext:
        return True

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('audio'):
        return True
    return False
    
def read_and_config_file(args, input_path, decode=0):
    """
    Reads and processes the input file or directory to extract audio file paths or configuration data.
    
    Parameters:
    args: The args
    input_path (str): Path to a file or directory containing audio data or file paths.
    decode (bool): If True (decode=1) for decoding, process the input as audio files directly (find .wav or .flac files) or from a .scp file.
                   If False (decode=0) for training, assume the input file contains lines with paths to audio files.
    
    Returns:
    processed_list (list): A list of processed file paths or a list of dictionaries containing input 
                           and optional condition audio paths.
    """
    processed_list = []  # Initialize list to hold processed file paths or configurations
    
    #The supported audio types are listed below (tested), but not limited to.
    file_ext = ["wav", "aac", "ac3", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wma", "webm"]
    
    if decode:
        # If decode is True, find audio files in a directory or single file
        if os.path.isdir(input_path):
            # Find all .wav files in the input directory
            processed_list = librosa.util.find_files(input_path, ext=file_ext)
        else:
            # If it's a single file and it's a .wav or .flac, add to processed list
            #if input_path.lower().endswith(".wav") or input_path.lower().endswith(".flac"):
            if is_audio_file(input_path):
                processed_list.append(input_path)
            else:
                # Read file paths from the input text file (one path per line)
                with open(input_path) as fid:
                    for line in fid:
                        path_s = line.strip().split()  # Split paths (space-separated)
                        processed_list.append(path_s[0])  # Add the first path (input audio path)
        return processed_list

    # If decode is False, treat the input file as a configuration file
    with open(input_path) as fid:
        for line in fid:
            tmp_paths = line.strip().split()  # Split paths (space-separated)
            if len(tmp_paths) == 2:
                # If two paths per line, treat the second as 'condition_audio'
                sample = {'inputs': tmp_paths[0], 'condition_audio': tmp_paths[1]}
            elif len(tmp_paths) == 1:
                # If only one path per line, treat it as 'inputs'
                sample = {'inputs': tmp_paths[0]}
            processed_list.append(sample)  # Append processed sample to list
    return processed_list

