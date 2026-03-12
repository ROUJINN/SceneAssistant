import time

import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")


def generate_image_with_zimage(
    prompt, seed=42, height=1024, width=1024, num_inference_steps=9
):
    # Optional: enable model compilation for faster large-batch inference.
    # pipe.transformer.compile()

    #  pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # 2. Generate Image
    time1 = time.time()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    time2 = time.time()
    print(f"Image generation time: {time2 - time1:.2f}s")
    # del pipe
    return image


if __name__ == "__main__":
    # prompt = "a 3d model of a bed."
    prompt = "Top-down orthographic view of a lush green grass texture, seamless and tileable."
    # prompt = (
    #     "Top-down orthographic view of a cement floor texture, seamless and tileable."
    # )
    seed = 0
    image = generate_image_with_zimage(prompt, seed)
    # image.save("images/test-bed-0.png")
    image.save("images/grass_texture.png")
