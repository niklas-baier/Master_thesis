#!/usr/bin/env python
# coding: utf-8

# # Introduction

# This notebook is a supplementary material for the Introduction Chapter of the [Hands-On Generative AI with Transformers and Diffusion Models book](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/).

# In[1]:


import diffusers
import huggingface_hub
import transformers

diffusers.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()


# ## Generating Images

# In[2]:


from genaibook.core import get_device

device = get_device()
print(f"Using device: {device}")


# In[ ]:


import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)


# In[4]:


prompt = "a photograph of an astronaut riding a horse"
pipe(prompt).images[0]


# In[ ]:


import torch
torch.manual_seed(0)


# ## Generating Text

# In[4]:


from transformers import pipeline

classifier = pipeline("text-classification", device=device)
classifier("This movie is disgustingly good !")


# In[5]:


from transformers import set_seed

# Setting the seed ensures we get the same results every time we run this code
set_seed(10)


# In[10]:


generator = pipeline("text-generation")
prompt = "It was a dark and stormy"
generator(prompt)[0]["generated_text"]


# ## Generating Sound Clips

# In[ ]:


pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)
data = pipe("electric rock solo, very intense")


# In[12]:


print(data)


# In[14]:


import IPython.display as ipd

display(ipd.Audio(data["audio"][0], rate=data["sampling_rate"]))


# In[ ]:




