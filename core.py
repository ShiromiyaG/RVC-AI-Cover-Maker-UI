import sys, os
import subprocess
import torch

from pedalboard import Pedalboard, Reverb
from pedalboard.io import AudioFile
from pydub import AudioSegment

now_dir = os.getcwd()
applio_dir = os.path.join(now_dir, "programs", "Applio")
sys.path.append(now_dir)
sys.path.append(applio_dir)

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
    all_models = models_vocals + karaoke_models + dereverb_models + deecho_models
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
):
    if devices == "-":
        force_cpu = True
    else:
        force_cpu = False
        devices = devices.replace("-", " ")
    # Vocals Separation
    model_info = get_model_info_by_name(vocal_model)
    download_file(
        model_info["model_url"],
        model_info["path"],
        "model.ckpt",
    )
    download_file(
        model_info["config_url"],
        model_info["path"],
        "config.yaml",
    )
    store_dir = os.path.join(now_dir, "audio_files", "vocals")
    os.makedirs(store_dir, exist_ok=True)
    command = [
        "python",
        os.path.join(
            now_dir, "programs", "Music-Source-Separation-Training", "inference.py"
        ),
        "--model_type",
        model_info["type"],
        "--config_path",
        model_info["config"],
        "--start_check_point",
        model_info["model"],
        "--input_file",
        input_audio_path,
        "--store_dir",
        store_dir,
        "--flac_file",
        "--pcm_type",
        "PCM_16",
        "--extract_instrumental",
    ]

    if force_cpu:
        command.append("--force_cpu")
    else:
        command.extend(["--device_ids", devices])
    print("Separating vocals")
    subprocess.run(command)

    # karaoke separation
    model_info = get_model_info_by_name(karaoke_model)
    store_dir = os.path.join(now_dir, "audio_files", "karaoke")
    os.makedirs(store_dir, exist_ok=True)
    vocals_path = os.path.join(now_dir, "audio_files", "vocals")
    input_file = search_with_word(vocals_path, "vocals")

    if input_file:
        input_file = os.path.join(vocals_path, input_file)
    print("Separating Bakcing vocals")
    if model_info["name"] == "Mel-Roformer Karaoke by aufr33 and viperx":
        download_file(
            model_info["model_url"],
            model_info["path"],
            "model.ckpt",
        )
        download_file(
            model_info["config_url"],
            model_info["path"],
            "config.yaml",
        )
        command = [
            "python",
            os.path.join(
                now_dir, "programs", "Music-Source-Separation-Training", "inference.py"
            ),
            "--model_type",
            model_info["type"],
            "--config_path",
            model_info["config"],
            "--start_check_point",
            model_info["model"],
            "--input_file",
            input_file,
            "--store_dir",
            store_dir,
            "--flac_file",
            "--pcm_type",
            "PCM_16",
            "--extract_instrumental",
        ]

        if force_cpu:
            command.append("--force_cpu")
        else:
            command.extend(["--device_ids", devices])

        subprocess.run(command)
    else:
        command = [
            "audio-separator",
            input_file,
            "--log_level",
            "warning",
            "--normalization",
            "1.0",
            "-m",
            model_info["full_name"],
            "--model_file_dir",
            os.path.join(now_dir, "models", "karaoke"),
            "--output_dir",
            store_dir,
        ]
        subprocess.run(command)

    # dereverb
    model_info = get_model_info_by_name(dereverb_model)
    store_dir = os.path.join(now_dir, "audio_files", "dereverb")
    os.makedirs(store_dir, exist_ok=True)
    karaoke_path = os.path.join(now_dir, "audio_files", "karaoke")
    input_file = search_with_word(karaoke_path, "karaoke") or search_with_word(
        karaoke_path, "Vocals"
    )

    if input_file:
        input_file = os.path.join(karaoke_path, input_file)
    print("Removing reverb")
    if (
        model_info["name"] == "BS-Roformer Dereverb by anvuew"
        or model_info["name"] == "MDX23C DeReverb by aufr33 and jarredou"
    ):
        download_file(
            model_info["model_url"],
            model_info["path"],
            "model.ckpt",
        )
        download_file(
            model_info["config_url"],
            model_info["path"],
            "config.yaml",
        )
        command = [
            "python",
            os.path.join(
                now_dir, "programs", "Music-Source-Separation-Training", "inference.py"
            ),
            "--model_type",
            model_info["type"],
            "--config_path",
            model_info["config"],
            "--start_check_point",
            model_info["model"],
            "--input_file",
            input_file,
            "--store_dir",
            store_dir,
            "--flac_file",
            "--pcm_type",
            "PCM_16",
        ]

        if force_cpu:
            command.append("--force_cpu")
        else:
            command.extend(["--device_ids", devices])

        subprocess.run(command)
    else:
        command = [
            "audio-separator",
            input_file,
            "--log_level",
            "warning",
            "--normalization",
            "1.0",
            "-m",
            model_info["full_name"],
            "--model_file_dir",
            os.path.join(now_dir, "models", "dereverb"),
            "--output_dir",
            store_dir,
        ]
        subprocess.run(command)

    # deecho
    store_dir = os.path.join(now_dir, "audio_files", "deecho")
    os.makedirs(store_dir, exist_ok=True)
    if deecho:
        print("Removing echo")
        model_info = get_model_info_by_name(deecho_model)

        dereverb_path = os.path.join(now_dir, "audio_files", "dereverb")
        noreverb_file = search_with_word(dereverb_path, "noreverb") or search_with_word(
            dereverb_path, "No Reverb"
        )

        input_file = os.path.join(dereverb_path, noreverb_file)

        command = [
            "audio-separator",
            input_file,
            "--log_level",
            "warning",
            "--normalization",
            "1.0",
            "-m",
            model_info["full_name"],
            "--model_file_dir",
            os.path.join(now_dir, "models", "deecho"),
            "--output_dir",
            store_dir,
        ]
        subprocess.run(command)

    # denoise
    store_dir = os.path.join(now_dir, "audio_files", "denoise")
    os.makedirs(store_dir, exist_ok=True)
    if denoise:
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

        if model_info["name"] == "mel-denoise":
            download_file(model_info["model_url"], model_info["path"], "model.ckpt")
            download_file(model_info["config_url"], model_info["path"], "config.yaml")

            command = [
                "python",
                os.path.join(
                    now_dir,
                    "programs",
                    "Music-Source-Separation-Training",
                    "inference.py",
                ),
                "--model_type",
                model_info["type"],
                "--config_path",
                model_info["config"],
                "--start_check_point",
                model_info["model"],
                "--input_file",
                input_file,
                "--store_dir",
                store_dir,
                "--flac_file",
                "--pcm_type",
                "PCM_16",
            ]

            if force_cpu:
                command.append("--force_cpu")
            else:
                command.extend(["--device_ids", devices])

            subprocess.run(command)
        else:
            command = [
                "audio-separator",
                "--input_file",
                input_file,
                "--log_level",
                "warning",
                "--normalization",
                "1.0",
                "-m",
                model_info["full_name"],
                "--model_file_dir",
                os.path.join(now_dir, "models", "denoise"),
                "--output_dir",
                store_dir,
            ]
            subprocess.run(command)

    # RVC
    denoise_path = os.path.join(now_dir, "audio_files", "denoise")
    deecho_path = os.path.join(now_dir, "audio_files", "deecho")
    dereverb_path = os.path.join(now_dir, "audio_files", "dereverb")
    denoise_audio = search_with_word(denoise_path, "No Noise") or search_with_word(
        denoise_path, "other"
    )
    deecho_audio = search_with_word(deecho_path, "No Echo")

    dereverb = search_with_word(dereverb_path, "No Reverb") or search_with_word(
        dereverb_path, "noreverb"
    )
    final_path = os.path.join(
        now_dir, "audio_files", "denoise", denoise_audio or deecho_audio or dereverb
    )

    store_dir = os.path.join(now_dir, "audio_files", "rvc")
    os.makedirs(store_dir, exist_ok=True)
    command = [
        "python",
        os.path.join(now_dir, "programs", "Applio", "core.py"),
        "infer",
        "--f0up_key",
        str(pitch),
        "--filter_radius",
        str(filter_radius),
        "--index_rate",
        str(index_rate),
        "--rms_mix_rate",
        str(rms_mix_rate),
        "--protect",
        str(protect),
        "--split_audio",
        split_audio,
        "--index_path",
        index_path,
        "--pth_path",
        model_path,
        "--input_path",
        final_path,
        "--output_path",
        store_dir,
        "--f0method",
        pitch_extract,
        "--f0autotune",
        autotune,
        "--hop_length",
        str(hop_lenght),
        "--export_format",
        export_format_rvc,
        "--embedder_model",
        embedder_model,
    ]
    os.chdir(os.path.join(now_dir, "programs", "Applio"))
    print("Making RVC inference")
    subprocess.run(command)
    os.chdir(now_dir)
    # post process
    if reverb:
        add_audio_effects(
            get_last_modified_file(os.path.join(now_dir, "audio_files", "rvc")),
            reverb_room_size,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_damping,
            reverb_width,
            output_path,
            os.path.join(
                now_dir, "audio_files", "rvc", os.path.basename(input_audio_path)
            ),
        )

    # merge audios
    store_dir = os.path.join(now_dir, "audio_files", "final")
    os.makedirs(store_dir, exist_ok=True)

    vocals_path = os.path.join(now_dir, "audio_files", "rvc")
    karaoke_path = os.path.join(now_dir, "audio_files", "karaoke")

    vocals_file = get_last_modified_file(vocals_path)
    karaoke_file = search_with_word(karaoke_path, "Instrumental") or search_with_word(
        karaoke_path, "instrumental"
    )

    final_output_path = os.path.join(
        now_dir,
        "audio_files",
        "final",
        f"{os.path.basename(input_audio_path)}_final",
    )
    print("Merging audios")
    return (
        f"Audio file {os.path.basename(input_audio_path)} converted with success",
        merge_audios(
            get_last_modified_file(os.path.join(now_dir, "audio_files", "rvc")),
            vocals_file,
            karaoke_file,
            final_output_path,
            vocals_volume,
            instrumentals_volume,
            backing_vocals_volume,
            export_format_final,
        ),
    )


def download_model(link):
    command = [
        "python",
        os.path.join(now_dir, "programs", "Applio", "core.py"),
        "download",
        "--model_link",
        link,
    ]
    os.chdir(os.path.join(now_dir, "programs", "Applio"))
    subprocess.run(command)
    os.chdir(now_dir)
    last_folder = get_last_modified_folder(
        os.path.join(now_dir, "programs", "Applio", "logs")
    )
    basename = os.path.basename(last_folder)
    os.rename(last_folder, os.path.join(now_dir, "logs", basename))
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
