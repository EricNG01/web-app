import torch
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from transformers import AutoProcessor, AutoModel
from PIL import Image, UnidentifiedImageError
# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "runwayml/stable-diffusion-v1-5"

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# prompt = "Generate a detailed, photorealistic image of a stovetop kettle boiling water. The kettle is metallic and shiny, with steam pouring out of its spout. The setting is a cozy kitchen with a wooden countertop and tiled backsplash. The scene should capture the moment when the water is at a rolling boil, emphasizing the steam and the kettle's reflective surface"
prompt = "Create an image of a kettle which is boiling water."
negative_prompt = "NSFW, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (extra legs:1.331), (fused fingers:1.5), (too many fingers:1.5), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit, bad hands, missing fingers, (((extra arms and legs))),"

image = pipe(
    prompt=prompt, 
    negative_prompt=negative_prompt,
    num_inference_steps=100, 
    guidance_scale=8,
    height=768, width=768
).images[0]
image.save("pp.png")


# # load pickScore model
# device = "cuda"
# processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
# model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

# processor = AutoProcessor.from_pretrained(processor_name_or_path)
# model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

# def calc_probs(prompt, images):
    
#     # preprocess
#     image_inputs = processor(
#         images=images,
#         padding=True,
#         truncation=True,
#         max_length=200,
#         return_tensors="pt",
#     ).to(device)
    
#     text_inputs = processor(
#         text=prompt,
#         padding=True,
#         truncation=True,
#         max_length=200,
#         return_tensors="pt",
#     ).to(device)


#     with torch.no_grad():
#         # embed
#         image_embs = model.get_image_features(**image_inputs)
#         image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
#         text_embs = model.get_text_features(**text_inputs)
#         text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
#         # score
#         scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
#         # get probabilities if you have multiple images to choose from
#         probs = torch.softmax(scores, dim=-1)  
    
#     return probs.cpu().tolist()
# def index_of_kth_largest(nums, k):
#     return np.argsort(nums)[-k]
# pick = []
# pick.append(Image.open("./toast.png"))
# pick.append(Image.open("./toaster.png"))

# pick[index_of_kth_largest(calc_probs(prompt, pick), 1)].show()

# image = pipe(
#     prompt=prompt, 
#     negative_prompt=negative_prompt,
#     num_inference_steps=300, 
#     guidance_scale=8,
#     height=512, width=512
# ).images[0]

# image.save("512_300steps.png")

# # load both base & refiner
# base = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# base.to("cuda")
# # base.enable_model_cpu_offload()
# refiner = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# # refiner.to("cuda")
# refiner.enable_model_cpu_offload()

# # Define how many steps and what % of steps to be run on each experts (80/20) here
# n_steps = 100
# high_noise_frac = 0.9

# # prompt - what(all required objects/stuffs), where(place/location), how(instruction)
# prompt = "uncooked spaghetti in a pot of boiling water, stove, kitchen, close-up, photorealistic, best quality"
# negative_prompt = "(((anime))), ((illustration)), cartoon, animation"


# # run both experts
# image = base(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=n_steps,
#     denoising_end=high_noise_frac,
#     output_type="latent",
#     height=768, # divisible by 8
#     width=768   # divisible by 8
# ).images
# image = refiner(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]


