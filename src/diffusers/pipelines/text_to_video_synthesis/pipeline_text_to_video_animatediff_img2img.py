from typing import Callable, Optional, Union, List, Dict, Any
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import torch
import PIL
import numpy as np

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from .pipeline_output import TextToVideoSDPipelineOutput
from ..stable_diffusion import StableDiffusionImg2ImgPipeline
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ...models import AutoencoderKL, UNet3DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils.torch_utils import randn_tensor
from ...utils import PIL_INTERPOLATION, logging, replace_example_docstring


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import UNet3DConditionModel, TextToVideoAnimateDiffPipeline 
        >>> from diffusers.utils import export_to_video

        >>> unet = UNet3DConditionModel.from_pretrained("itsadarshms/animatediff-v2-stable-diffusion-1.5", subfolder="unet", torch_dtype=torch.float16, use_safetensors=True)
        >>> pipe = TextToVideoAnimateDiffPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> prompt = "portrait photo of a old warrior chief"
        >>> video_frames = pipe(prompt).frames
        >>> video_path = export_to_video(video_frames)
        >>> video_path
        ```
"""

"""
Copied from https://github.com/neggles/animatediff-cli/blob/main/src/animatediff/pipelines/context.py
- ordered_halving
- uniform
- get_total_steps
- get_context_params
"""

def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


# I have absolutely no idea how this works and I don't like that.
def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]


def get_total_steps(
    scheduler,
    timesteps: list[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )


def get_context_params(
    length: int,
    context: Optional[int] = None,
    overlap: Optional[int] = None,
    stride: Optional[int] = None,
):
    if context is None:
        context = min(length, 16)
    if overlap is None:
        overlap = context // 2
    if stride is None:
        stride = 4
    return context, overlap, stride


def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    # This code is modified from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images


def preprocess_video(video, num_frames=None):
    supported_formats = (np.ndarray, torch.Tensor, PIL.Image.Image)

    if isinstance(video, supported_formats):
        video = [video]
    elif not (isinstance(video, list) and all(isinstance(i, supported_formats) for i in video)):
        raise ValueError(
            f"Input is in incorrect format: {[type(i) for i in video]}. Currently, we only support {', '.join(supported_formats)}"
        )

    if isinstance(video[0], (PIL.Image.Image, np.ndarray)):
        w, h = video[0].size if isinstance(video[0], PIL.Image.Image) else (video[0].shape[1], video[0].shape[0])
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        video = [np.array(frame.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]) if isinstance(frame, PIL.Image.Image) else PIL.Image.fromarray(frame).resize((w, h), resample=PIL_INTERPOLATION["lanczos"])) for frame in video]
        
        video = np.concatenate(video, axis=0) if video[0].ndim == 5 else np.stack(video, axis=0)
        if video.dtype == np.uint8:
            video = np.array(video).astype(np.float32) / 255.0

        if video.ndim == 4:
            video = video[None, ...]

        video = torch.from_numpy(video.transpose(0, 4, 1, 2, 3))

    elif isinstance(video[0], torch.Tensor):
        video = torch.cat(video, axis=0) if video[0].ndim == 5 else torch.stack(video, axis=0)

        # don't need any preprocess if the video is latents
        channel = video.shape[1]
        if channel == 4:
            return video

        # move channels before num_frames
        video = video.permute(0, 2, 1, 3, 4)

    # normalize video
    video = 2.0 * video - 1.0

    # Input is an image/single frame, repeat to num_frames
    if video.shape[2] == 1:
        num_frames = num_frames or 16
        video = video.repeat(1, 1, num_frames, 1, 1)
    elif num_frames is not None and video.shape[2] != num_frames:
        logger.warning(f"Input is a video of {video.shape[2]} frames, num_frames parameter will be ignored.")

    return video


class VideoToVideoAnimateDiffPipeline(StableDiffusionImg2ImgPipeline):
    r"""
    Pipeline for AnimateDiff text-to-video generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer (`CLIPTokenizer`):
            A [`~transformers.CLIPTokenizer`] to tokenize text.
        unet ([`UNet3DConditionModel`]):
            A [`UNet3DConditionModel`] to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`CLIPImageProcessor`]):
            A [`CLIPImageProcessor`] to extract features from generated images; used as inputs to the `safety_checker`.
    """

    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor, requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)
    
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        image = self.image_processor.postprocess(image, output_type="pt")

        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.cpu().float()
        return video
    
    def prepare_latents(self, video, timestep, batch_size, dtype, device, generator=None):
        video = video.to(device=device, dtype=dtype)

        # change from (b, c, f, h, w) -> (b * f, c, w, h)
        bsz, channel, frames, width, height = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

        if video.shape[1] == 4:
            init_latents = video
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(video[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(video).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `video` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        shape = init_latents.shape
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        latents = latents[None, :].reshape((bsz, frames, latents.shape[1]) + latents.shape[2:]).permute(0, 2, 1, 3, 4)

        return latents
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        video: Union[List[np.ndarray], List[PIL.Image.Image], List[torch.FloatTensor], PIL.Image.Image, np.ndarray, torch.FloatTensor] = None,
        strength: float = 0.6,
        num_frames: int = None,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        context_frames: Optional[int] = None,
        context_stride: Optional[int] = None,
        context_overlap: Optional[int] = None,
        latents_device: Optional[str] = None
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            video (`List[np.ndarray]` or `torch.FloatTensor`):
                `video` frames or tensor representing a video batch to be used as the starting point for the process.
                Can also accept video latents as `image`, if passing latents directly, it will not be encoded again.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `video`. Must be between 0 and 1. `video` is used as a
                starting point, adding more noise to it the larger the `strength`. The number of denoising steps
                depends on the amount of noise initially added. When `strength` is 1, added noise is maximum and the
                denoising process runs for the full number of iterations specified in `num_inference_steps`. A value of
                1 essentially ignores `video`.
            num_frames (`int`, *optional*, defaults to 16):
                The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                amounts to 2 seconds of video. This is only applicable when the input video has only a single frame/is an image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] instead
                of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        Examples:

        Returns:
            [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Preprocess video
        video = preprocess_video(video, num_frames)

        # 16 frames is max reliable number for one-shot mode, so we use sequential mode for longer videos
        num_frames = video.shape[2]
        sequential_mode = num_frames is not None and num_frames > 16
        context_frames, context_overlap, context_stride = get_context_params(num_frames, context_frames, context_overlap, context_stride)

        if latents_device is None:
            latents_device = torch.device("cpu") if sequential_mode else device
        else:
            latents_device = torch.device(latents_device)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            video,
            latent_timestep,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            latents_device, # keep latents on cpu for sequential mode
            generator
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.5 - Infinite context loop shenanigans
        context_scheduler = uniform
        total_steps = get_total_steps(
            context_scheduler,
            timesteps,
            num_inference_steps,
            latents.shape[2],
            context_frames,
            context_stride,
            context_overlap,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=total_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros(
                    (latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    device=latents.device,
                    dtype=latents.dtype,
                )
                counter = torch.zeros(
                    (1, 1, latents.shape[2], 1, 1), device=latents.device, dtype=latents.dtype
                )

                for context in context_scheduler(
                    i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap
                ):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (
                        latents[:, :, context]
                        .to(device)
                        .repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    )
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    pred = self.unet(
                        latent_model_input.to(self.unet.device, self.unet.dtype),
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]

                    pred = pred.to(dtype=latents.dtype, device=latents.device)
                    noise_pred[:, :, context] = noise_pred[:, :, context] + pred
                    counter[:, :, context] = counter[:, :, context] + 1
                    progress_bar.update()

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents.to(latents_device), **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            return TextToVideoSDPipelineOutput(frames=latents)

        video_tensor = self.decode_latents(latents.to(device))

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)