# coding: utf-8
__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import argparse
import time
import librosa
from tqdm import tqdm
import sys
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils import demix, get_model_from_config

import warnings

warnings.filterwarnings("ignore")


class Args:
    def __init__(
        self,
        input_file,
        store_dir,
        model_type,
        extract_instrumental,
        disable_detailed_pbar,
        flac_file,
        pcm_type,
        use_tta,
    ):
        self.input_file = input_file
        self.model_type = model_type
        self.store_dir = store_dir
        self.extract_instrumental = extract_instrumental
        self.disable_detailed_pbar = disable_detailed_pbar
        self.flac_file = flac_file
        self.pcm_type = pcm_type
        self.use_tta = use_tta


def run_file(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    if not os.path.isfile(args.input_file):
        print("File not found: {}".format(args.input_file))
        return

    instruments = config.training.instruments.copy()
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    print("Starting processing track: ", args.input_file)
    try:
        mix, sr = librosa.load(args.input_file, sr=44100, mono=False)
    except Exception as e:
        print("Cannot read track: {}".format(args.input_file))
        print("Error message: {}".format(str(e)))
        return

    # Convert mono to stereo if needed
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_orig = mix.copy()
    if "normalize" in config.inference:
        if config.inference["normalize"] is True:
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std

    if args.use_tta:
        # orig, channel inverse, polarity inverse
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1.0 * mix.copy()]
    else:
        track_proc_list = [mix.copy()]

    full_result = []
    for mix in track_proc_list:
        waveforms = demix(
            config, model, mix, device, pbar=verbose, model_type=args.model_type
        )
        full_result.append(waveforms)

    # Average all values in single dict
    waveforms = full_result[0]
    for i in range(1, len(full_result)):
        d = full_result[i]
        for el in d:
            if i == 2:
                waveforms[el] += -1.0 * d[el]
            elif i == 1:
                waveforms[el] += d[el][::-1].copy()
            else:
                waveforms[el] += d[el]
    for el in waveforms:
        waveforms[el] = waveforms[el] / len(full_result)

    # Create a new `instr` in instruments list, 'instrumental'
    if args.extract_instrumental:
        instr = "vocals" if "vocals" in instruments else instruments[0]
        instruments.append("instrumental")
        # Output "instrumental", which is an inverse of 'vocals' or the first stem in list if 'vocals' absent
        waveforms["instrumental"] = mix_orig - waveforms[instr]

    for instr in instruments:
        estimates = waveforms[instr].T
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                estimates = estimates * std + mean
        file_name, _ = os.path.splitext(os.path.basename(args.input_file))
        if args.flac_file:
            output_file = os.path.join(args.store_dir, f"{file_name}_{instr}.flac")
            subtype = "PCM_16" if args.pcm_type == "PCM_16" else "PCM_24"
            sf.write(output_file, estimates, sr, subtype=subtype)
        else:
            output_file = os.path.join(args.store_dir, f"{file_name}_{instr}.wav")
            sf.write(output_file, estimates, sr, subtype="FLOAT")

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_file(
    model_type,
    config_path,
    start_check_point,
    input_file,
    store_dir,
    device_ids,
    extract_instrumental,
    disable_detailed_pbar,
    flac_file,
    pcm_type,
    use_tta,
):
    print("Using device: ", device_ids)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(model_type, config_path)
    if start_check_point != "":
        print("Start from checkpoint: {}".format(start_check_point))
        if model_type == "htdemucs":
            state_dict = torch.load(
                start_check_point, map_location=device_ids, weights_only=False
            )
            # Fix for htdemucs pretrained models
            if "state" in state_dict:
                state_dict = state_dict["state"]
        else:
            state_dict = torch.load(
                start_check_point, map_location=device_ids, weights_only=True
            )
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    if type(device_ids) == list and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model = model.to(device_ids)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))
    args = Args(
        input_file,
        store_dir,
        model_type,
        extract_instrumental,
        disable_detailed_pbar,
        flac_file,
        pcm_type,
        use_tta,
    )

    run_file(model, args, config, device_ids, verbose=True)
