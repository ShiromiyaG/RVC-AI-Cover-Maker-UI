import sys, os
import subprocess
import torch
from functools import lru_cache
import shutil
from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment
from audio_separator.separator import Separator
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)
from programs.applio_code.rvc.infer.infer import VoiceConverter
from programs.applio_code.rvc.lib.tools.model_download import model_download_pipeline
from programs.music_separation_code.inference import proc_file

models_vocals = [
    {
        "name": "Mel-Roformer by KimberleyJSN",
        "path": os.path.join(now_dir, "models", "mel-vocals"),
        "model": os.path.join(now_dir, "models", "mel-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-vocals", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml",
        "model_url": "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt",
    },
    {
        "name": "BS-Roformer by ViperX",
        "path": os.path.join(now_dir, "models", "bs-vocals"),
        "model": os.path.join(now_dir, "models", "bs-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "bs-vocals", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "model_url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    },
    {
        "name": "MDX23C",
        "path": os.path.join(now_dir, "models", "mdx23c-vocals"),
        "model": os.path.join(now_dir, "models", "mdx23c-vocals", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-vocals", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_vocals_mdx23c.yaml",
        "model_url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt",
    },
]

karaoke_models = [
    {
        "name": "Mel-Roformer Karaoke by aufr33 and viperx",
        "path": os.path.join(now_dir, "models", "mel-kara"),
        "model": os.path.join(now_dir, "models", "mel-kara", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-kara", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/config_mel_band_roformer_karaoke.yaml",
        "model_url": "https://huggingface.co/shiromiya/audio-separation-models/resolve/main/mel_band_roformer_karaoke_aufr33_viperx/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    },
    {"name": "UVR-BVE", "full_name": "UVR-BVE-4B_SN-44100-1.pth"},
]

denoise_models = [
    {
        "name": "Mel-Roformer Denoise Normal by aufr33",
        "path": os.path.join(now_dir, "models", "mel-denoise"),
        "model": os.path.join(now_dir, "models", "mel-denoise", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-denoise", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    },
    {
        "name": "Mel-Roformer Denoise Aggressive by aufr33",
        "path": os.path.join(now_dir, "models", "mel-denoise-aggr"),
        "model": os.path.join(now_dir, "models", "mel-denoise-aggr", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mel-denoise-aggr", "config.yaml"),
        "type": "mel_band_roformer",
        "config_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/model_mel_band_roformer_denoise.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    },
    {
        "name": "UVR Denoise",
        "full_name": "UVR-DeNoise.pth",
    },
]

dereverb_models = [
    {
        "name": "MDX23C DeReverb by aufr33 and jarredou",
        "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
        "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
        "type": "mdx23c",
        "config_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/config_dereverb_mdx23c.yaml",
        "model_url": "https://huggingface.co/jarredou/aufr33_jarredou_MDXv3_DeReverb/resolve/main/dereverb_mdx23c_sdr_6.9096.ckpt",
    },
    {
        "name": "BS-Roformer Dereverb by anvuew",
        "path": os.path.join(now_dir, "models", "mdx23c-dereveb"),
        "model": os.path.join(now_dir, "models", "mdx23c-dereveb", "model.ckpt"),
        "config": os.path.join(now_dir, "models", "mdx23c-dereveb", "config.yaml"),
        "type": "bs_roformer",
        "config_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.yaml",
        "model_url": "https://huggingface.co/anvuew/deverb_bs_roformer/resolve/main/deverb_bs_roformer_8_384dim_10depth.ckpt",
    },
    {
        "name": "UVR-Deecho-Dereverb",
        "full_name": "UVR-DeEcho-DeReverb.pth",
    },
    {
        "name": "MDX Reverb HQ by FoxJoy",
        "full_name": "Reverb_HQ_By_FoxJoy.onnx",
    },
]

deecho_models = [
    {
        "name": "UVR-Deecho-Normal",
        "full_name": "UVR-De-Echo-Normal.pth",
    },
    {
        "name": "UVR-Deecho-Agggressive",
        "full_name": "UVR-De-Echo-Aggressive.pth",
    },
]


@lru_cache(maxsize=None)
def import_voice_converter():
    from programs.applio_code.rvc.infer.infer import VoiceConverter

    return VoiceConverter()


@lru_cache(maxsize=1)
def get_config():
    from programs.applio_code.rvc.configs.config import Config

    return Config()


def download_file(url, path, filename):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        print(f"File '{filename}' already exists at '{path}'.")
        return

    try:
        response = torch.hub.download_url_to_file(url, file_path)
        print(f"File '{filename}' downloaded successfully")
    except Exception as e:
        print(f"Error downloading file '{filename}' from '{url}': {e}")


def get_model_info_by_name(model_name):
    all_models = models_vocals + karaoke_models + dereverb_models + deecho_models + denoise_models
    for model in all_models:
        if model["name"] == model_name:
            return model
    return None


def get_last_modified_file(pasta):
    if not os.path.isdir(pasta):
        raise NotADirectoryError(f"{pasta} is not a valid directory.")
    arquivos = [f for f in os.listdir(pasta) if os.path.isfile(os.path.join(pasta, f))]
    if not arquivos:
        return None
    return max(arquivos, key=lambda x: os.path.getmtime(os.path.join(pasta, x)))


def search_with_word(folder, word):
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"{folder} is not a valid directory.")
    file_with_word = [file for file in os.listdir(folder) if word in file]
    return file_with_word[-1] if file_with_word else None


def get_last_modified_folder(path):
    directories = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ]
    if not directories:
        return None
    last_modified_folder = max(directories, key=os.path.getmtime)
    return last_modified_folder


def add_audio_effects(
    audio_path,
    reverb_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
    reverb_width,
    output_path,
):
    board = Pedalboard([])
    board.append(
        Reverb(
            room_size=reverb_size,
            dry_level=reverb_dry,
            wet_level=reverb_wet,
            damping=reverb_damping,
            width=reverb_width,
        )
    )
    with AudioFile(audio_path) as f:
        with AudioFile(output_path, "w", f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
    return output_path


def merge_audios(
    vocals_path,
    inst_path,
    backing_path,
    output_path,
    main_gain,
    inst_gain,
    backing_Vol,
    output_format,
):
    main_vocal_audio = AudioSegment.from_file(vocals_path, format="flac") + main_gain
    instrumental_audio = AudioSegment.from_file(inst_path, format="flac") + inst_gain
    backing_vocal_audio = (
        AudioSegment.from_file(backing_path, format="flac") + backing_Vol
    )
    combined_audio = main_vocal_audio.overlay(
        instrumental_audio.overlay(backing_vocal_audio)
    )
    combined_audio.export(output_path, format=output_format)
    return output_path


def full_inference_program(
    model_path,
    index_path,
    input_audio_path,
    output_path,
    export_format_rvc,
    split_audio,
    autotune,
    vocal_model,
    karaoke_model,
    dereverb_model,
    deecho,
    deecho_model,
    denoise,
    denoise_model,
    reverb,
    vocals_volume,
    instrumentals_volume,
    backing_vocals_volume,
    export_format_final,
    devices,
    pitch,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    pitch_extract,
    hop_lenght,
    reverb_room_size,
    reverb_damping,
    reverb_wet_gain,
    reverb_dry_gain,
    reverb_width,
    embedder_model,
    delete_audios,
    use_tta,
    batch_size,
):
    # Configuração de devices
    if devices == "-":
        device = "cpu"
    else:
        devices = devices.split("-")
        if type(devices) == list:
            device = f'cuda:{devices[0]}'
        else:
            device = f'cuda:{devices}'
    # Vocals Separation
    model_info = get_model_info_by_name(vocal_model)
    model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
    if not os.path.exists(model_ckpt_path):
        download_file(
            model_info["model_url"],
            model_info["path"],
            "model.ckpt",
        )
    config_json_path = os.path.join(model_info["path"], "config.yaml")
    if not os.path.exists(config_json_path):
        download_file(
            model_info["config_url"],
            model_info["path"],
            "config.yaml",
        )
    store_dir = os.path.join(now_dir, "audio_files", "vocals")
    inst_dir = os.path.join(now_dir, "audio_files", "instrumentals")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(inst_dir, exist_ok=True)
    if search_with_word(store_dir, "vocals") != None:
        print("Vocals already separated"),
    else:
        print("Separating vocals")
        proc_file(
            model_type=model_info["type"],
            config_path=model_info["config"],
            start_check_point=model_info["model"],
            input_file=input_audio_path,
            store_dir=store_dir,
            device_ids=devices,
            device=device,
            extract_instrumental=True,
            disable_detailed_pbar=False,
            flac_file=True,
            pcm_type="PCM_16",
            use_tta=use_tta,
        )
        os.rename(
            os.path.join(store_dir, search_with_word(store_dir, "instrumental")),
            os.path.join(inst_dir, "instrumentals.flac"),
        )
    inst_file = os.path.join(inst_dir, "instrumentals.flac")

    # karaoke separation
    model_info = get_model_info_by_name(karaoke_model)
    store_dir = os.path.join(now_dir, "audio_files", "karaoke")
    os.makedirs(store_dir, exist_ok=True)
    vocals_path = os.path.join(now_dir, "audio_files", "vocals")
    input_file = search_with_word(vocals_path, "vocals")
    if (
        search_with_word(store_dir, "karaoke") != None
        or search_with_word(store_dir, "Vocals") != None
    ):
        print("Backing vocals already separated")
    else:
        if input_file:
            input_file = os.path.join(vocals_path, input_file)
        print("Separating Bakcing vocals")
        if model_info["name"] == "Mel-Roformer Karaoke by aufr33 and viperx":
            model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
            if not os.path.exists(model_ckpt_path):
                download_file(
                    model_info["model_url"],
                    model_info["path"],
                    "model.ckpt",
                )
            config_json_path = os.path.join(model_info["path"], "config.yaml")
            if not os.path.exists(config_json_path):
                download_file(
                    model_info["config_url"],
                    model_info["path"],
                    "config.yaml",
                )
            proc_file(
                model_type=model_info["type"],
                config_path=model_info["config"],
                start_check_point=model_info["model"],
                input_file=input_file,
                store_dir=store_dir,
                device_ids=devices,
                device=device,
                extract_instrumental=True,
                disable_detailed_pbar=False,
                flac_file=True,
                pcm_type="PCM_16",
                use_tta=use_tta,
            )
        else:
            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "karaoke"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=store_dir,
                vr_params= {
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                }
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)

    # dereverb
    model_info = get_model_info_by_name(dereverb_model)
    store_dir = os.path.join(now_dir, "audio_files", "dereverb")
    os.makedirs(store_dir, exist_ok=True)
    karaoke_path = os.path.join(now_dir, "audio_files", "karaoke")
    input_file = search_with_word(karaoke_path, "karaoke") or search_with_word(
        karaoke_path, "Vocals"
    )
    if (
        search_with_word(store_dir, "noreverb") != None
        or search_with_word(store_dir, "No Reverb") != None
    ):
        print("Reverb already removed")
    else:
        if input_file:
            input_file = os.path.join(karaoke_path, input_file)
        print("Removing reverb")
        if (
            model_info["name"] == "BS-Roformer Dereverb by anvuew"
            or model_info["name"] == "MDX23C DeReverb by aufr33 and jarredou"
        ):
            model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
            if not os.path.exists(model_ckpt_path):
                download_file(
                    model_info["model_url"],
                    model_info["path"],
                    "model.ckpt",
                )
            config_json_path = os.path.join(model_info["path"], "config.yaml")
            if not os.path.exists(config_json_path):
                download_file(
                    model_info["config_url"],
                    model_info["path"],
                    "config.yaml",
                )
            proc_file(
                model_type=model_info["type"],
                config_path=model_info["config"],
                start_check_point=model_info["model"],
                input_file=input_file,
                store_dir=store_dir,
                device_ids=devices,
                device=device,
                extract_instrumental=False,
                disable_detailed_pbar=False,
                flac_file=True,
                pcm_type="PCM_16",
                use_tta=use_tta,
            )
        else:
            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "dereverb"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=store_dir,
                vr_params= {
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                }
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)

    # deecho
    store_dir = os.path.join(now_dir, "audio_files", "deecho")
    os.makedirs(store_dir, exist_ok=True)
    if deecho:
        if search_with_word(store_dir, "No Echo") != None:
            print("Echo already removed")
        else:
            print("Removing echo")
            model_info = get_model_info_by_name(deecho_model)

            dereverb_path = os.path.join(now_dir, "audio_files", "dereverb")
            noreverb_file = search_with_word(
                dereverb_path, "noreverb"
            ) or search_with_word(dereverb_path, "No Reverb")

            input_file = os.path.join(dereverb_path, noreverb_file)

            separator = Separator(
                model_file_dir=os.path.join(now_dir, "models", "deecho"),
                log_level=logging.WARNING,
                normalization_threshold=1.0,
                output_format="flac",
                output_dir=store_dir,
                vr_params= {
                    "batch_size": batch_size,
                    "enable_tta": use_tta,
                }
            )
            separator.load_model(model_filename=model_info["full_name"])
            separator.separate(input_file)

    # denoise
    store_dir = os.path.join(now_dir, "audio_files", "denoise")
    os.makedirs(store_dir, exist_ok=True)
    if denoise:
        if search_with_word(store_dir, "No Noise") != None or search_with_word(
            store_dir, "dry" != None
        ):
            print("Noise already removed")
        else:
            model_info = get_model_info_by_name(denoise_model)
            print("Removing noise")
            input_file = (
                os.path.join(
                    now_dir,
                    "audio_files",
                    "deecho",
                    search_with_word(
                        os.path.join(now_dir, "audio_files", "deecho"), "No Echo"
                    ),
                )
                if deecho
                else os.path.join(
                    now_dir,
                    "audio_files",
                    "dereverb",
                    search_with_word(
                        os.path.join(now_dir, "audio_files", "dereverb"), "noreverb"
                    )
                    or search_with_word(
                        os.path.join(now_dir, "audio_files", "dereverb"), "No Reverb"
                    ),
                )
            )

            if model_info["name"] == "Mel-Roformer Denoise Normal by aufr33" or model_info["mame"] == "Mel-Roformer Denoise Aggressive by aufr33":
                model_ckpt_path = os.path.join(model_info["path"], "model.ckpt")
                if not os.path.exists(model_ckpt_path):
                    download_file(
                        model_info["model_url"],
                        model_info["path"],
                        "model.ckpt",
                    )
                config_json_path = os.path.join(model_info["path"], "config.yaml")
                if not os.path.exists(config_json_path):
                    download_file(
                        model_info["config_url"],
                        model_info["path"],
                        "config.yaml"
                    )
                proc_file(
                    model_type=model_info["type"],
                    config_path=model_info["config"],
                    start_check_point=model_info["model"],
                    input_file=input_file,
                    store_dir=store_dir,
                    device_ids=devices,
                    device=device,
                    extract_instrumental=False,
                    disable_detailed_pbar=False,
                    flac_file=True,
                    pcm_type="PCM_16",
                    use_tta=use_tta,
                )
            else:
                separator = Separator(
                    model_file_dir=os.path.join(now_dir, "models", "denoise"),
                    log_level=logging.WARNING,
                    normalization_threshold=1.0,
                    output_format="flac",
                    output_dir=store_dir,
                    vr_params= {
                        "batch_size": batch_size,
                        "enable_tta": use_tta,
                    }
                )
                separator.load_model(model_filename=model_info["full_name"])
                separator.separate(input_file)

    # RVC
    denoise_path = os.path.join(now_dir, "audio_files", "denoise")
    deecho_path = os.path.join(now_dir, "audio_files", "deecho")
    dereverb_path = os.path.join(now_dir, "audio_files", "dereverb")

    denoise_audio = search_with_word(denoise_path, "No Noise") or search_with_word(
        denoise_path, "dry"
    )
    deecho_audio = search_with_word(deecho_path, "No Echo")
    dereverb = search_with_word(dereverb_path, "No Reverb") or search_with_word(
        dereverb_path, "noreverb"
    )

    if denoise_audio:
        final_path = os.path.join(now_dir, "audio_files", "denoise", denoise_audio)
    elif deecho_audio:
        final_path = os.path.join(now_dir, "audio_files", "deecho", deecho_audio)
    elif dereverb:
        final_path = os.path.join(now_dir, "audio_files", "dereverb", dereverb)
    else:
        final_path = None

    store_dir = os.path.join(now_dir, "audio_files", "rvc")
    os.makedirs(store_dir, exist_ok=True)
    print("Making RVC inference")
    inference_vc = import_voice_converter()
    inference_vc.convert_audio(
        audio_input_path=final_path,
        audio_output_path=output_path,
        model_path=model_path,
        index_path=index_path,
        embedder_model=embedder_model,
        pitch=pitch,
        f0_file=None,
        f0_method=pitch_extract,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=rms_mix_rate,
        protect=protect,
        split_audio=split_audio,
        f0_autotune=autotune,
        hop_length=hop_lenght,
        export_format=export_format_rvc,
        embedder_model_custom=None,
    )
    # post process
    if reverb:
        add_audio_effects(
            os.path.join(
                now_dir,
                "audio_files",
                "rvc",
                get_last_modified_file(os.path.join(now_dir, "audio_files", "rvc")),
            ),
            reverb_room_size,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_damping,
            reverb_width,
            os.path.join(
                now_dir, "audio_files", "rvc", os.path.basename(input_audio_path)
            ),
        )

    # merge audios
    store_dir = os.path.join(now_dir, "audio_files", "final")
    os.makedirs(store_dir, exist_ok=True)

    vocals_path = os.path.join(now_dir, "audio_files", "rvc")
    vocals_file = get_last_modified_file(os.path.join(now_dir, "audio_files", "rvc"))
    vocals_file = os.path.join(vocals_path, vocals_file)

    karaoke_path = os.path.join(now_dir, "audio_files", "karaoke")
    karaoke_file = search_with_word(karaoke_path, "Instrumental") or search_with_word(
        karaoke_path, "instrumental"
    )
    karaoke_file = os.path.join(karaoke_path, karaoke_file)
    final_output_path = os.path.join(
        now_dir,
        "audio_files",
        "final",
        f"{os.path.basename(input_audio_path).split('.')[0]}_final.{export_format_final.lower()}",
    )
    print("Merging audios")
    result = merge_audios(
        vocals_file,
        inst_file,
        karaoke_file,
        final_output_path,
        vocals_volume,
        instrumentals_volume,
        backing_vocals_volume,
        export_format_final,
    )
    print("Audios merged!")
    if delete_audios:
        main_directory = os.path.join(now_dir, "audio_files")
        folder_to_keep = "final"
        for folder_name in os.listdir(main_directory):
            folder_path = os.path.join(main_directory, folder_name)
            if os.path.isdir(folder_path) and folder_name != folder_to_keep:
                shutil.rmtree(folder_path)
    return (
        f"Audio file {os.path.basename(input_audio_path)} converted with success",
        result,
    )


def download_model(link):
    model_download_pipeline(link)
    return "Model downloaded with success"


def download_music(link):
    command = [
        "yt-dlp",
        "-x",
        "--output",
        os.path.join(now_dir, "audio_files", "original_files", "%(title)s.%(ext)s"),
        link,
    ]
    subprocess.run(command)
    return "Music downloaded with success"
