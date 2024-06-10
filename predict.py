# Prediction interface for Cog ⚙️
# https://cog.run/python

import subprocess
import time
from cog import BasePredictor, Input, Path
import argparse
import os
import cv2
import numpy as np
import torch
import torchaudio.functional
import torchvision.io
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection, Wav2Vec2Model, Wav2Vec2Processor
from imageio_ffmpeg import get_ffmpeg_exe

from modules import (
    UNet2DConditionModel,
    UNet3DConditionModel,
    VKpsGuider,
    AudioProjection,
)
from pipelines import VExpressPipeline
from pipelines.utils import draw_kps_image, save_video
from pipelines.utils import retarget_kps

MODEL_CACHE = "model_ckpts"


def download_weights(url: str, dest: str) -> None:
    # NOTE WHEN YOU EXTRACT SPECIFY THE PARENT FOLDER
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.use_pget_and_download_weights()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        vae_path = "./model_ckpts/sd-vae-ft-mse/"
        audio_encoder_path = "./model_ckpts/wav2vec2-base-960h/"

        self.vae = AutoencoderKL.from_pretrained(vae_path).to(
            dtype=self.dtype, device=self.device
        )
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path).to(
            dtype=self.dtype, device=self.device
        )
        self.audio_processor = Wav2Vec2Processor.from_pretrained(audio_encoder_path)

        unet_config_path = "./model_ckpts/stable-diffusion-v1-5/unet/config.json"
        reference_net_path = "./model_ckpts/v-express/reference_net.pth"
        denoising_unet_path = "./model_ckpts/v-express/denoising_unet.pth"
        v_kps_guider_path = "./model_ckpts/v-express/v_kps_guider.pth"
        audio_projection_path = "./model_ckpts/v-express/audio_projection.pth"
        motion_module_path = "./model_ckpts/v-express/motion_module.pth"

        inference_config_path = "./inference_v2.yaml"
        self.scheduler = self.get_scheduler(inference_config_path)
        self.reference_net = self.load_reference_net(
            unet_config_path, reference_net_path, self.dtype, self.device
        )
        self.denoising_unet = self.load_denoising_unet(
            inference_config_path,
            unet_config_path,
            denoising_unet_path,
            motion_module_path,
            self.dtype,
            self.device,
        )
        self.v_kps_guider = self.load_v_kps_guider(
            v_kps_guider_path, self.dtype, self.device
        )
        self.audio_projection = self.load_audio_projection(
            audio_projection_path,
            self.dtype,
            self.device,
            inp_dim=self.denoising_unet.config.cross_attention_dim,
            mid_dim=self.denoising_unet.config.cross_attention_dim,
            out_dim=self.denoising_unet.config.cross_attention_dim,
            inp_seq_len=2 * (2 * 2 + 1),
            out_seq_len=2 * 2 + 1,
        )

        if is_xformers_available():
            self.reference_net.enable_xformers_memory_efficient_attention()
            self.denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

        self.generator = torch.manual_seed(42)
        self.pipeline = VExpressPipeline(
            vae=self.vae,
            reference_net=self.reference_net,
            denoising_unet=self.denoising_unet,
            v_kps_guider=self.v_kps_guider,
            audio_processor=self.audio_processor,
            audio_encoder=self.audio_encoder,
            audio_projection=self.audio_projection,
            scheduler=self.scheduler,
        ).to(dtype=self.dtype, device=self.device)

        self.app = FaceAnalysis(
            providers=[
                (
                    "CUDAExecutionProvider"
                    if self.device.type == "cuda"
                    else "CPUExecutionProvider"
                )
            ],
            provider_options=[{"device_id": 0}] if self.device.type == "cuda" else [],
            root="./model_ckpts/insightface_models/",
        )
        self.app.prepare(ctx_id=0, det_size=(512, 512))

    def use_pget_and_download_weights(self):

        # Create directories if they don't exist
        os.makedirs(MODEL_CACHE, exist_ok=True)

        # Model files and base URLs
        model_files = [
            "insightface_models.tar",
            "stable-diffusion-v1-5.tar",
            "wav2vec2-base-960h.tar",
            "sd-vae-ft-mse.tar",
            "v-express.tar",
        ]

        base_url = (
            f"https://weights.replicate.delivery/default/V-Express/{MODEL_CACHE}/"
        )

        # Download model files
        for model_file in model_files:
            url = base_url + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

    def load_reference_net(self, unet_config_path, reference_net_path, dtype, device):
        reference_net = UNet2DConditionModel.from_config(unet_config_path).to(
            dtype=dtype, device=device
        )
        reference_net.load_state_dict(
            torch.load(reference_net_path, map_location="cpu"), strict=False
        )
        return reference_net

    def load_denoising_unet(
        self,
        inference_config_path,
        unet_config_path,
        denoising_unet_path,
        motion_module_path,
        dtype,
        device,
    ):
        inference_config = OmegaConf.load(inference_config_path)
        denoising_unet = UNet3DConditionModel.from_config_2d(
            unet_config_path,
            unet_additional_kwargs=inference_config.unet_additional_kwargs,
        ).to(dtype=dtype, device=device)
        denoising_unet.load_state_dict(
            torch.load(denoising_unet_path, map_location="cpu"), strict=False
        )
        denoising_unet.load_state_dict(
            torch.load(motion_module_path, map_location="cpu"), strict=False
        )
        return denoising_unet

    def load_v_kps_guider(self, v_kps_guider_path, dtype, device):
        v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256)).to(
            dtype=dtype, device=device
        )
        v_kps_guider.load_state_dict(torch.load(v_kps_guider_path, map_location="cpu"))
        return v_kps_guider

    def load_audio_projection(
        self,
        audio_projection_path,
        dtype,
        device,
        inp_dim,
        mid_dim,
        out_dim,
        inp_seq_len,
        out_seq_len,
    ):
        audio_projection = AudioProjection(
            dim=mid_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=out_seq_len,
            embedding_dim=inp_dim,
            output_dim=out_dim,
            ff_mult=4,
            max_seq_len=inp_seq_len,
        ).to(dtype=dtype, device=device)
        audio_projection.load_state_dict(
            torch.load(audio_projection_path, map_location="cpu")
        )
        return audio_projection

    def get_scheduler(self, inference_config_path):
        inference_config = OmegaConf.load(inference_config_path)
        scheduler_kwargs = OmegaConf.to_container(
            inference_config.noise_scheduler_kwargs
        )
        scheduler = DDIMScheduler(**scheduler_kwargs)
        return scheduler

    def predict(
        self,
        reference_image: Path = Input(
            description="Path to the reference image that will be used as the base for the generated video."
        ),
        driving_audio: Path = Input(
            description="Path to the audio file that will be used to drive the motion in the generated video."
        ),
        driving_video: Path = Input(
            description="Path to the video file that will be used to extract the head motion. If not provided, the generated video will use the motion based on the selected motion_mode.",
            default=None,
        ),
        motion_mode: str = Input(
            description="Mode for generating the head motion in the output video.",
            choices=["standard", "gentle", "normal", "fast"],
            default="fast",
        ),
        reference_attention_weight: float = Input(
            description="Amount of attention to pay to the reference image vs. the driving motion. Higher values will make the generated video adhere more closely to the reference image. Range: 0.0 to 1.0",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        audio_attention_weight: float = Input(
            description="Amount of attention to pay to the driving audio vs. the reference image. Higher values will make the generated video's motion more closely match the driving audio. Range: 0.0 to 10.0",
            ge=0.0,
            le=10.0,
            default=3.0,
        ),
        num_inference_steps: int = Input(
            description="Number of diffusion steps to perform during generation. More steps will generally produce better quality results but will take longer to run. Range: 1 to 100",
            ge=1,
            le=100,
            default=25,
        ),
        image_width: int = Input(
            description="Width of the generated video frames.",
            ge=64,
            le=2048,
            default=512,
        ),
        image_height: int = Input(
            description="Height of the generated video frames.",
            ge=64,
            le=2048,
            default=512,
        ),
        frames_per_second: float = Input(
            description="Frame rate of the generated video.",
            ge=1,
            le=60,
            default=30.0,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion model. Higher values will result in the generated video following the driving motion and audio more closely.",
            ge=1,
            le=20,
            default=3.5,
        ),
        num_context_frames: int = Input(
            description="Number of context frames to use for motion estimation.",
            ge=1,
            le=24,
            default=12,
        ),
        context_stride: int = Input(
            description="Stride of the context frames.",
            ge=1,
            le=10,
            default=1,
        ),
        context_overlap: int = Input(
            description="Number of overlapping frames between context windows.",
            ge=0,
            le=24,
            default=4,
        ),
        num_audio_padding_frames: int = Input(
            description="Number of audio frames to pad on each side of the driving audio.",
            ge=0,
            le=10,
            default=2,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # Reassign input parameters to their original variable names
        output_path = "./output_video.mp4"
        reference_image_path = str(reference_image)
        audio_path = str(driving_audio)
        target_video_path = str(driving_video)
        standard_audio_sampling_rate = 16000
        fps = frames_per_second
        context_frames = num_context_frames
        num_pad_audio_frames = num_audio_padding_frames
        generator = torch.manual_seed(seed)

        # Reassign motion_mode to retarget_strategy with the original values
        if motion_mode == "standard":
            retarget_strategy = "fix_face"
        elif motion_mode == "gentle":
            retarget_strategy = "offset_retarget"
        elif motion_mode == "normal":
            retarget_strategy = "offset_retarget"
        elif motion_mode == "fast":
            retarget_strategy = "naive_retarget"
        else:
            raise ValueError(f"Unsupported motion mode '{motion_mode}'.")

        # Get kps_sequence if target_video_path (i.e. driving_video) is provided
        # We'll use the video to drive the head motion
        if target_video_path is not None:
            # Extract keypoints and audio from the driving video using the script
            kps_path = "temp_kps.pth"
            temp_audio_path = "temp_audio.mp3"
            if os.path.exists(kps_path):
                os.remove(kps_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            os.system(
                f"python scripts/extract_kps_sequence_and_audio.py --video_path {target_video_path} --kps_sequence_save_path {kps_path} --audio_save_path {temp_audio_path}"
            )
            kps_sequence = torch.load(kps_path)
            audio_waveform, audio_sampling_rate = torchaudio.load(temp_audio_path)
        else:
            kps_path = None
            kps_sequence = None
            _, audio_waveform, meta_info = torchvision.io.read_video(
                str(audio_path), pts_unit="sec"
            )
            audio_sampling_rate = meta_info["audio_fps"]

        reference_image = Image.open(reference_image_path).convert("RGB")
        reference_image = reference_image.resize((image_width, image_height))

        reference_image_for_kps = cv2.imread(reference_image_path)
        reference_image_for_kps = cv2.resize(
            reference_image_for_kps, (image_width, image_height)
        )
        reference_kps = self.app.get(reference_image_for_kps)[0].kps[:3]

        _, audio_waveform, meta_info = torchvision.io.read_video(
            audio_path, pts_unit="sec"
        )
        audio_sampling_rate = meta_info["audio_fps"]
        print(
            f"Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}."
        )
        if audio_sampling_rate != standard_audio_sampling_rate:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform,
                orig_freq=audio_sampling_rate,
                new_freq=standard_audio_sampling_rate,
            )
        audio_waveform = audio_waveform.mean(dim=0)

        duration = audio_waveform.shape[0] / standard_audio_sampling_rate
        video_length = int(duration * fps)
        print(f"The corresponding video length is {video_length}.")

        if kps_path != "":
            assert os.path.exists(kps_path), f"{kps_path} does not exist"
            kps_sequence = torch.tensor(torch.load(kps_path))  # [len, 3, 2]
            print(f"The original length of kps sequence is {kps_sequence.shape[0]}.")
            kps_sequence = torch.nn.functional.interpolate(
                kps_sequence.permute(1, 2, 0), size=video_length, mode="linear"
            )
            kps_sequence = kps_sequence.permute(2, 0, 1)
            print(
                f"The interpolated length of kps sequence is {kps_sequence.shape[0]}."
            )

        if retarget_strategy == "fix_face":
            kps_sequence = torch.tensor([reference_kps] * video_length)
        elif retarget_strategy == "no_retarget":
            kps_sequence = kps_sequence
        elif retarget_strategy == "offset_retarget":
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
        elif retarget_strategy == "naive_retarget":
            kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
        else:
            raise ValueError(
                f"The retarget strategy {retarget_strategy} is not supported."
            )

        kps_images = []
        for i in range(video_length):
            kps_image = np.zeros_like(reference_image_for_kps)
            kps_image = draw_kps_image(kps_image, kps_sequence[i])
            kps_images.append(Image.fromarray(kps_image))

        vae_scale_factor = 8
        latent_height = image_height // vae_scale_factor
        latent_width = image_width // vae_scale_factor

        latent_shape = (1, 4, video_length, latent_height, latent_width)
        vae_latents = randn_tensor(
            latent_shape, generator=generator, device=self.device, dtype=self.dtype
        )

        video_latents = self.pipeline(
            vae_latents=vae_latents,
            reference_image=reference_image,
            kps_images=kps_images,
            audio_waveform=audio_waveform,
            width=image_width,
            height=image_height,
            video_length=video_length,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            context_frames=context_frames,
            context_stride=context_stride,
            context_overlap=context_overlap,
            reference_attention_weight=reference_attention_weight,
            audio_attention_weight=audio_attention_weight,
            num_pad_audio_frames=num_pad_audio_frames,
            generator=generator,
        ).video_latents

        video_tensor = self.pipeline.decode_latents(video_latents)
        if isinstance(video_tensor, np.ndarray):
            video_tensor = torch.from_numpy(video_tensor)

            save_video(video_tensor, audio_path, output_path, fps)

        return Path(output_path)
