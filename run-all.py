import hashlib
import pathlib
import sys
sys.path.append('./')


import os 
import cv2
import torch
import random
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import pydantic

from ip_adapter import IPAdapterXL



# global variable
MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32

# initialization
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "IP-Adapter/sdxl_models/image_encoder"
ip_ckpt = "IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"

controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[
            offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new
        ] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image



def create_image(image_pil,
                 input_image,
                 prompt,
                 n_prompt,
                 scale, 
                 control_scale, 
                 guidance_scale,
                 num_samples,
                 num_inference_steps,
                 seed,
                 target="Load only style blocks",
                 neg_content_prompt=None,
                 neg_content_scale=0):

    if target =="Load original IP-Adapter":
        # target_blocks=["blocks"] for original IP-Adapter
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["blocks"])
    elif target=="Load only style blocks":
        # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])
    elif target == "Load style+layout block":
        # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])
    
    if input_image is not None:
        input_image = resize_img(input_image, max_side=1024)
        cv_input_image = pil_to_cv2(input_image)
        detected_map = cv2.Canny(cv_input_image, 50, 200)
        canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))
    else:
        canny_map = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
        control_scale = 0

    if float(control_scale) == 0:
        canny_map = canny_map.resize((1024,1024))
    
    if len(neg_content_prompt) > 0 and neg_content_scale != 0:
        images = ip_model.generate(pil_image=image_pil,
                                prompt=prompt,
                                negative_prompt=n_prompt,
                                scale=scale,
                                guidance_scale=guidance_scale,
                                num_samples=num_samples,
                                num_inference_steps=num_inference_steps, 
                                seed=seed,
                                image=canny_map,
                                controlnet_conditioning_scale=float(control_scale),
                                neg_content_prompt=neg_content_prompt,
                                neg_content_scale=neg_content_scale
                                )
    else:
        images = ip_model.generate(pil_image=image_pil,
                                prompt=prompt,
                                negative_prompt=n_prompt,
                                scale=scale,
                                guidance_scale=guidance_scale,
                                num_samples=num_samples,
                                num_inference_steps=num_inference_steps, 
                                seed=seed,
                                image=canny_map,
                                controlnet_conditioning_scale=float(control_scale),
                                )
    return images

def create_simple(style_image, source_image, prompt, scale, control_scale):

    return create_image(
        image_pil=style_image,
        input_image=source_image,
        prompt=prompt,
        n_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        scale=scale,
        control_scale=control_scale,
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=20,
        # num_inference_steps=1,
        seed=42,
        target="Load only style blocks",
        neg_content_prompt="",
        neg_content_scale=0,
    )




style_image=Image.open("images\oliver-jeffers-1.jpg")

class Parameters(pydantic.BaseModel):
    style_image_path: str
    source_image_path: str | None = None
    prompt: str | None = None
    scale: float = 1.0
    control_scale: float = 0.5 # I think this is for how much like the original image it should be 0 is no original image, or not like the image
    seed: int = 42
    negative_prompt: str = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
    guidance_scale: int = 5
    num_samples: int = 1
    num_inference_steps: int = 20
    neg_content_prompt: str = ""
    neg_content_scale: int = 0

    def hash_properties(self) -> str:
        sha256 = hashlib.sha256()
        for key, value in sorted(self.model_dump().items()):
            sha256.update(f"{key}={value}".encode('utf-8'))
        hex_result = sha256.hexdigest()

        return hex_result[:8]
class Result(pydantic.BaseModel):
    parameters: Parameters
    result_image_path: str


def create_image_from_input(input: Parameters):
    style_image = Image.open(input.style_image_path)
    source_image = Image.open(input.source_image_path)
    source_image = resize_img(source_image, max_side=1024)

    return create_image(
        image_pil=style_image,
        input_image=source_image,
        prompt=input.prompt,
        n_prompt=input.negative_prompt,
        scale=input.scale,
        control_scale=input.control_scale,
        guidance_scale=input.guidance_scale,
        num_samples=input.num_samples,
        num_inference_steps=input.num_inference_steps,
        # num_inference_steps=1,
        seed=input.seed,
        target="Load only style blocks",
        neg_content_prompt=input.neg_content_prompt,
        neg_content_scale=input.neg_content_scale,
    )

prompts = [
    "A stingray swimming",
    "A stingray swimming in the ocean",
]


source_images_dir = "source-images"
output_images_dir = "output-images"

os.makedirs(source_images_dir, exist_ok=True)

def generate_all_images_no_prompt():
    to_generate : list[Parameters] = []

    source_images = []
    for img in os.listdir(source_images_dir):
        if img.endswith(('.png', '.jpg', '.jpeg')):
            # filename = "".join(pathlib.Path(generate_params.source_image_path).name.split(".")[:-1])
            to_generate.append(
                Parameters(
                    style_image_path="images\oliver-jeffers-1.jpg", 
                    source_image_path=os.path.join(source_images_dir, img)
                )
            )
    return to_generate

def custom_generate():
    to_generate : list[Parameters] = []

    to_generate.append(
        Parameters(
            style_image_path="images\\oliver-jeffers-1.jpg",
            source_image_path="source-images\\crayfish.jpg",
            prompt="A crayfish",
            scale=1.0,
            control_scale=0.5,
            seed=42,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            guidance_scale=5,
            num_samples=1,
            num_inference_steps=20,
            neg_content_prompt="",
            neg_content_scale=0
        )
    )
    to_generate.append(
        Parameters(
            style_image_path="images\\oliver-jeffers-1.jpg",
            source_image_path="source-images\\Ray Charles.jpg",
            prompt="Ray charles playing a piano, in black and white",
            scale=1.0,
            control_scale=0.5,
            seed=42,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            guidance_scale=5,
            num_samples=1,
            num_inference_steps=20,
            neg_content_prompt="",
            neg_content_scale=0
        )
    )
    to_generate.append(
        Parameters(
            style_image_path="images\\oliver-jeffers-1.jpg",
            source_image_path="source-images\\Eames Chair.jpg",
            prompt="Eames chair, with black leather and hardwood",
            scale=1.0,
            control_scale=0.5,
            seed=42,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            guidance_scale=5,
            num_samples=1,
            num_inference_steps=20,
            neg_content_prompt="",
            neg_content_scale=0
        )
    )
    return to_generate

to_generate = custom_generate()

style_images = ["images\\oliver-jeffers-1.jpg", "images\\beach-oliver-jeffers.webp"]

to_generate = [
    Parameters(
        style_image_path="images\\oliver-jeffers-1.jpg",
        source_image_path="source-images\\Crayons.png",
        prompt="A box of crayons. Each crayon should be a unique color. Vibrant colors.",
        scale=1.0,
        control_scale=0.5,
        seed=42,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=20,
        neg_content_prompt="",
        neg_content_scale=0
    ),
        Parameters(
        style_image_path="images\\beach-oliver-jeffers.webp",
        source_image_path="source-images\\Crayons.png",
        prompt="A box of crayons. Each crayon should be a unique color. Vibrant colors.",
        scale=1.0,
        control_scale=0.5,
        seed=42,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=20,
        neg_content_prompt="",
        neg_content_scale=0
    ),


    Parameters(
        style_image_path="images\\oliver-jeffers-1.jpg",
        source_image_path="source-images\\Rainbow.jpg",
        prompt="A rainbow. Vibrant colors.",
        scale=1.0,
        control_scale=0.5,
        seed=42,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=20,
        neg_content_prompt="",
        neg_content_scale=0
    ),
        Parameters(
        style_image_path="images\\beach-oliver-jeffers.webp",
        source_image_path="source-images\\Crayons.png",
        prompt="A rainbow. Vibrant colors.",
        scale=1.0,
        control_scale=0.5,
        seed=42,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        guidance_scale=5,
        num_samples=1,
        num_inference_steps=20,
        neg_content_prompt="",
        neg_content_scale=0
    )
]


print(len(to_generate))

for generate_params in to_generate:
    filename = "".join(pathlib.Path(generate_params.source_image_path).name.split(".")[:-1])
    hash_value = format(hash(generate_params) & 0xFFFFFFFFFFFFFFFF, '016x')[:8]
    result_image_path = f"{filename}-{hash_value}"
    result_json = f"{filename}-{hash_value}.json"
    result_path_json : pathlib.Path = pathlib.Path(output_images_dir).joinpath(result_json)
    result = Result(parameters=generate_params, result_image_path=result_image_path)
    if result_path_json.exists():
        print(f"Already has result in {result_json} + {result_image_path}")
        continue
    print(f"Generating {generate_params}")

    result_images : list[Image.Image] = create_image_from_input(generate_params)
    for idx, img in enumerate(result_images):
        img.save(os.path.join(output_images_dir, result_image_path + "-" + str(idx) + ".jpg"), "JPEG")

    result_path_json.write_text(result.model_dump_json(indent=2))

        # source_images.append((img, Image.open(os.path.join(source_images_dir, img))))


# to_generate : list[Parameters] = []

# source_images = []
# for img in os.listdir(source_images_dir):
#     if img.endswith(('.png', '.jpg', '.jpeg')):
#         to_generate.append(
#             Parameters(
#                 style_image_path="images\oliver-jeffers-1.jpg", 
#                 source_image_path=os.path.join(source_images_dir, img)
#             )
#         )
#         # source_images.append((img, Image.open(os.path.join(source_images_dir, img))))

# for source_image in source_images:
#     source_image_name, source_image = source_image  # Extract the Image object from the tuple
#     source_image = resize_img(source_image, max_side=1024)

#     result_images : list[Image.Image] = create_simple(style_image, source_image=source_image, prompt=None, scale=1.0, control_scale=0.5)

#     output_dir = "output_images"
#     os.makedirs(output_dir, exist_ok=True)

#     for idx, img in enumerate(result_images):
#         img.save(os.path.join(output_dir, source_image_name.split(".")[0] + ".jpg"), "JPEG")