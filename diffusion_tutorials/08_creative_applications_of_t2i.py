#!/usr/bin/env python
# coding: utf-8

# # Creative Applications of Text-To-Image Models
# 
# This notebook is a supplementary material for the Creative Applications of Text-To-Image Models of the [Hands-On Generative AI with Transformers and Diffusion Models book](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/). This notebook includes:
# 
# * The code from the book
# * Additional examples
# * Exercise solutions

# ## Image-to-Image

# In[1]:


import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

from genaibook.core import get_device

device = get_device()

# Load the pipeline
img2img_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)


# In[2]:


# Move the pipeline to the device
# Alternatively, img2img_pipeline.enable_model_cpu_offload()
img2img_pipeline.to(device)


# In[3]:


from genaibook.core import image_grid, load_image, SampleURL

# Load the image
url = SampleURL.ToyAstronauts
init_image = load_image(url)
generator = torch.Generator(device=device).manual_seed(1)

prompt = "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k"

# Pass the prompt and the image through the pipeline
image = img2img_pipeline(prompt, image=init_image, generator=generator, strength=0.7).images[0]
image_grid([init_image, image], rows=1, cols=2)


# In[4]:


del img2img_pipeline
torch.cuda.empty_cache()


# ## Inpainting

# In[5]:


from diffusers import StableDiffusionXLInpaintPipeline

# Load the pipeline
inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)

img_url = SampleURL.DogBenchImage
mask_url = SampleURL.DogBenchMask

init_image = load_image(img_url, size=(1024, 1024))
mask_image = load_image(mask_url, size=(1024, 1024))

# Pass images and prompt through the pipeline
prompt = "A majestic tiger sitting on a bench"
image = inpaint_pipeline(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=50,
    strength=0.80,
    width=init_image.size[0],
    heigth=init_image.size[1],
).images[0]

image_grid([init_image, mask_image, image], rows=1, cols=3)


# In[6]:


del inpaint_pipeline
torch.cuda.empty_cache()


# ## Prompt Weighting and Image Editing

# ### Prompt Weighting and Merging 

# In[7]:


from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)


# In[8]:


from compel import Compel, ReturnedEmbeddingsType

# Use the penultimate CLIP layer as it is more expressive
embeddings_type = (
    ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
)
compel = Compel(
    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
    returned_embeddings_type=embeddings_type,
    requires_pooled=[False, True],
)


# In[9]:


from diffusers.utils import make_image_grid

# Prepare the prompts
prompts = []
prompts.append("a humanoid robot eating pasta")
prompts.append(
    "a humanoid+++ robot eating pasta"
)  # make its humanoid characteristics a bit more pronounced
prompts.append(
    '["a humanoid robot eating pasta", "a van gogh painting"].and(0.8, 0.2)'
)  # make it van gogh!

images = []
for prompt in prompts:
    # Use the same seed across generations
    generator = torch.Generator(device=device).manual_seed(1)

    # The compel library returns both the conditioning vectors 
    # and the pooled prompt embeds
    conditioning, pooled = compel(prompt)

    # We pass the conditioning and pooled prompt embeds to the pipeline
    image = pipeline(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        num_inference_steps=30,
        generator=generator,
    ).images[0]
    images.append(image)
image_grid(images, rows=1, cols=3)


# In[10]:


del pipeline
del conditioning
del pooled
del compel
torch.cuda.empty_cache()


# #### Editing Diffusion Images with Semantic Guidance

# In[11]:


from diffusers import SemanticStableDiffusionPipeline

semantic_pipeline = SemanticStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, variant="fp16"
).to(device)


# In[12]:


seed = 7667


# In[13]:


generator = torch.Generator(device=device).manual_seed(seed)
out = semantic_pipeline(
    prompt="a photo of the face of a man",
    negative_prompt="low quality, deformed",
    generator=generator,
)
out.images[0]


# In[14]:


generator = torch.Generator(device=device).manual_seed(seed)
out = semantic_pipeline(
    prompt="a photo of the face of a man",
    negative_prompt="low quality, deformed",
    editing_prompt="smiling, smile",
    edit_guidance_scale=4,
    edit_warmup_steps=10,
    edit_threshold=0.99,
    edit_momentum_scale=0.3,
    edit_mom_beta=0.6,
    reverse_editing_direction=False,
    generator=generator,
)
out.images[0]


# In[15]:


generator = torch.Generator(device=device).manual_seed(seed)
out = semantic_pipeline(
    prompt="a photo of the face of a man",
    negative_prompt="low quality, deformed",
    editing_prompt="glasses, wearing glasses",
    reverse_editing_direction=False,
    edit_warmup_steps=10,
    edit_guidance_scale=4,
    edit_threshold=0.99,
    edit_momentum_scale=0.3,
    edit_mom_beta=0.6,
    generator=generator,
)
out.images[0]


# In[16]:


generator = torch.Generator(device=device).manual_seed(seed)
out = semantic_pipeline(
    prompt="a photo of the face of a man",
    negative_prompt="low quality, deformed",
    editing_prompt=[
        "smiling, smile",
        "glasses, wearing glasses",
    ],
    reverse_editing_direction=[False, False],
    edit_warmup_steps=[10, 10],
    edit_guidance_scale=[6, 6],
    edit_threshold=[0.99, 0.99],
    edit_momentum_scale=0.3,
    edit_mom_beta=0.6,
    generator=generator,
)
out.images[0]


# In[17]:


del semantic_pipeline
torch.cuda.empty_cache()


# ## Real Image Editing via Inversion
# 
# ### Editing with LEDITS++

# In[18]:


from diffusers import LEditsPPPipelineStableDiffusion

# Load the model as usual
pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to(device)

image = load_image(SampleURL.ManInGlasses)

# Invert the image, gradually adding noise to it so
# it can be denoised with modified directions,
# effectively providing an edit
pipe.invert(image=image, num_inversion_steps=50, skip=0.2)

# Edit the image with an editing prompt
edited_image = pipe(
    editing_prompt=["glasses"],
    # tell the model to remove the glasses by editing the direction
    reverse_editing_direction=[True],
    edit_guidance_scale=[1.5],
    edit_threshold=[0.95],
).images[0]

image_grid([image, edited_image], rows=1, cols=2)


# In[19]:


del pipe
torch.cuda.empty_cache()


# ### Real Image Editing via Instruction Fine-Tuning

# In[20]:


from diffusers import (
    EDMEulerScheduler,
    StableDiffusionXLInstructPix2PixPipeline,
)
from huggingface_hub import hf_hub_download

edit_file = hf_hub_download(
    repo_id="stabilityai/cosxl", filename="cosxl_edit.safetensors"
)

# from_single_file loads a diffusion model from a single diffusers file
pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file, num_in_channels=8, is_cosxl_edit=True, torch_dtype=torch.float16
)

# The model was trained so that the EDMEulerScheduler
# is the correct noise scheduler for denoising
pipe_edit.scheduler = EDMEulerScheduler(
    sigma_min=0.002,
    sigma_max=120.0,
    sigma_data=1.0,
    prediction_type="v_prediction",
    sigma_schedule="exponential",
)
pipe_edit.to(device)

prompt = "make it a cloudy day"
image = load_image(SampleURL.Mountain)
edited_image = pipe_edit(
    prompt=prompt, image=image, num_inference_steps=20
).images[0]

image_grid([image, edited_image], rows=1, cols=2)


# In[21]:


del pipe_edit
torch.cuda.empty_cache()


# ## ControlNet

# In[22]:


from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)

controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
# controlnet_pipeline.enable_model_cpu_offload()  # Optional, saves VRAM
controlnet_pipeline = controlnet_pipeline.to(device)


# In[23]:


from controlnet_aux import MidasDetector
from PIL import Image

original_image = load_image(SampleURL.WomanSpeaking, size=(1024, 1024))

# loads the MiDAS depth detector model
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

# Apply MiDAS depth detection
processed_image_midas = midas(original_image).resize(
    (1024, 1024), Image.BICUBIC
)


# In[24]:


image = controlnet_pipeline(
    "A colorful, ultra-realistic masked super hero singing a song",
    image=processed_image_midas,
    controlnet_conditioning_scale=0.4,
    num_inference_steps=30,
).images[0]
image_grid([original_image, processed_image_midas, image], rows=1, cols=3)


# In[25]:


del controlnet
del controlnet_pipeline
torch.cuda.empty_cache()


# ## Image Prompting and Image Variations
# 
# ### Image Variations

# In[26]:


from diffusers import StableDiffusionXLPipeline

sdxl_base_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
)
sdxl_base_pipeline.to(device)

# We load the IP Adapter too
sdxl_base_pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
)

# We can set the scale of how strong we
# want our IP Adapter to impact our overall result
sdxl_base_pipeline.set_ip_adapter_scale(0.8)

image = load_image(SampleURL.ItemsVariation)
original_image = image.resize((1024, 1024))

# Create the image variation
generator = torch.Generator(device=device).manual_seed(1)
variation_image = sdxl_base_pipeline(
    prompt="",
    ip_adapter_image=original_image,
    num_inference_steps=25,
    generator=generator,
).images

image_grid([original_image, variation_image[0]], rows=1, cols=2)


# In[27]:


del sdxl_base_pipeline
torch.cuda.empty_cache()


# ### Image Prompting
# 
# #### Style Transfer

# In[28]:


# We load the model and the IP Adapter, just as before
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to(device)

# Load the IP Adapter into the model
pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
)

# We are applying the IP Adapter only to the mid block,
# which is where it should be mapped to the style in SDXL
scale = {"up": {"block_0": [0.0, 1.0, 0.0]}}
pipeline.set_ip_adapter_scale(scale)

image = load_image(SampleURL.Mamoeiro)
original_image = image.resize((1024, 1024))

# Run inference to generate the stylized image
generator = torch.Generator(device=device).manual_seed(0)
variation_image = pipeline(
    prompt="a cat inside of a box",
    ip_adapter_image=original_image,
    num_inference_steps=25,
    generator=generator,
).images

image_grid([original_image, variation_image[0]], rows=1, cols=2)


# In[29]:


del pipeline
torch.cuda.empty_cache()


# #### Additional Controls
# 

# In[30]:


controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
)

# Load the ControlNet pipeline
controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
controlnet_pipeline.to(device)

# Load the IP Adapter
controlnet_pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
)
# We are applying the IP Adapter only to the mid block,
# which is where it should be mapped to the style in SDXL
scale = {
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
controlnet_pipeline.set_ip_adapter_scale(scale)

# Load the original image
original_image = load_image(SampleURL.WomanSpeaking)
original_image = original_image.resize((1024, 1024))

# Load the style image
style_image = load_image(SampleURL.Mamoeiro)
style_image = style_image.resize((1024, 1024))

# Apply the MiDAS depth estimation
processed_image_midas = midas(original_image).resize(
    (1024, 1024), Image.BICUBIC
)

image = controlnet_pipeline(
    "A masked super hero singing a song",
    image=processed_image_midas,
    ip_adapter_image=style_image,
    controlnet_conditioning_scale=0.5,
).images[0]
image_grid(
    [original_image, style_image, processed_image_midas, image], rows=1, cols=4
)


# In[31]:


del controlnet
del controlnet_pipeline
torch.cuda.empty_cache()


# ## Solutions
# 
# A big part of learning is putting your knowledge into practice. We strongly suggest not looking at the answers before taking a serious stab at it. Scroll down for the answers.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Exercises
# 
# **1. Explain how inpainting differs from image-to-image transformation and provide an example of a practical application.**
# 
# Inpainting fills in or replaces specific masked areas of an image, while image-to-image transforms the entire image. Inpainting can be used to remove unwanted objects from images, such as removing a person from a photo. An example of image-to-image is converting a daytime scene into a nighttime scene.
# 

# **2. How can prompt weighting help overcome the limitations of the diffusion models?**
# 
# By providing fine-grained control over the prompt interpretation. 
#  
# Users can emphasize or de-emphasize certain elements in the final image by assigning different weights to various parts of the prompt. This is particularly useful when the model might be biased towards certain interpretations or when specific aspects of the prompt need more attention. For example, if a user wants to generate an image of "a red car in a forest" but the model keeps focusing too much on the forest, they could increase the weight of "red car" in the prompt. This fine-grained control helps users achieve their desired results more accurately, especially in cases where the default behavior of the model might not produce the intended outcome.

# **3. What are the key differences between Prompt-to-Prompt editing and SEGA?**
# 
# Prompt-to-Prompt 
# * Focuses on modifying generated images by tweaking the input prompts while preserving the structure or overall composition of the original image.
# * Directly modifies cross-attention layers and leverages text embeddings to guide the editing process.
# 
# SEGA
# * Aims to guide the generation process by amplifying or suppressing specific semantic concepts.
# * Adjusts the latent space guidance signal to prioritize semantic features during generation.

# **4. How does ControlNet enhance the capabilities of diffusion models? Give examples of conditions that can be used with ControlNet.**
# 
# ControlNet enhances the capabilities of diffusion models by allowing for additional control over the image generation process using various types of input conditions. It does this by training a separate neural network that learns to interpret these conditions and guide the main diffusion model accordingly. This allows for much more precise control over the generated images, beyond what's possible with text prompts alone. Some examples of conditions that can be used with ControlNet include:
# 
# a. Edge maps or sketches, allowing the model to generate images that follow specific outlines or structures.
# b. Pose estimation data, which can guide the model to generate images of people or animals in specific poses.
# c. Depth maps, enabling the model to generate images with specific 3D structures or perspectives.
# d. Segmentation maps, allowing for control over the layout and composition of generated images.
# e. Normal maps, which can guide the model in generating images with specific surface details and textures.

# **5. What is "Inversion" in the context of text-to-image models, and what does it allow us to do?**
# 
# It's a process to map real images into the latent space of the model for editing. The real image is "inverted" back into noise mapped into the latent space which allows for perfect reconstruction and more excitingly, for editing purposes.

# 
