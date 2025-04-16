
import os
import pydantic
import pathlib
import hashlib

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

    def __hash__(self, *args, **kwargs):
        print("=====")
        for key, value in sorted(self.model_dump().items()):
            print(f"key={key}, value={value}")
        print("=====")

        tuple_result = tuple((key,value) for key,value in sorted(self.model_dump().items()))
        print(tuple_result)


        return hash(tuple_result)


# source_images_dir = "source-images"
# output_images_dir = "output-images"

# os.makedirs(source_images_dir, exist_ok=True)

# to_generate : list[Parameters] = []


# source_images = []
# for img in os.listdir(source_images_dir):
#     print(img)
#     if img.endswith(('.png', '.jpg', '.jpeg')):
#         to_generate.append(
#             Parameters(
#                 style_image_path="images\oliver-jeffers-1.jpg", 
#                 source_image_path=os.path.join(source_images_dir, img)
#             )
#         )


# print(len(to_generate))

# for generate_params in to_generate:
#     filename = "".join(pathlib.Path(generate_params.source_image_path).name.split(".")[:-1])
#     hash_value = format(hash(generate_params) & 0xFFFFFFFFFFFFFFFF, '016x')[:8]
#     result_filename = f"{filename}-{hash_value}.jpg"
#     print(result_filename)
#         # source_images.append((img, Image.open(os.path.join(source_images_dir, img))))


a = Parameters(
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
    )

filename = "".join(pathlib.Path(a.source_image_path).name.split(".")[:-1])
hash_value = a.hash_properties()
print(hash_value)