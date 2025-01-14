#!/usr/bin/env python
# coding: utf-8

# # Appendix C. End-to-End Retrieval-Augmented Generation
# 

# This notebook is a supplementary material for the Appendix C of the [Hands-On Generative AI with Transformers and Diffusion Models book](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/).

# ## Processing the Data
# 

# In[12]:


import urllib.request

# Define the file name and URL
file_name = "The-AI-Act.pdf"
url = "https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf"

# Download the file
urllib.request.urlretrieve(url, file_name)
print(f"{file_name} downloaded successfully.")


# In[ ]:


pip install langchain_community pypdf langchain-text-splitters


# In[13]:


from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_name)
docs = loader.load()
print(len(docs))


# In[14]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
chunks = text_splitter.split_documents(docs)
print(len(chunks))


# In[15]:


chunked_text = [chunk.page_content for chunk in chunks]
chunked_text[404]


# ### Embedding the Documents
# 

# In[16]:


from sentence_transformers import SentenceTransformer, util

sentences = ["I'm happy", "I'm full of happiness"]
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Compute embedding for both sentences
embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
embedding_2 = model.encode(sentences[1], convert_to_tensor=True)


# In[17]:


embedding_1.shape


# In[18]:


util.pytorch_cos_sim(embedding_1, embedding_2)


# In[19]:


embedding_1 @ embedding_2


# In[20]:


import torch

torch.dot(embedding_1, embedding_2)


# In[23]:


chunk_embeddings = model.encode(chunked_text, convert_to_tensor=True)


# In[24]:


chunk_embeddings.shape


# ## Retrieval

# In[25]:


def search_documents(query, top_k=5):
    # Encode the query into a vector
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity between the query and all document chunks
    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)

    # Get the top k most similar chunks
    top_k_indices = similarities[0].topk(top_k).indices

    # Retrieve the corresponding document chunks
    results = [chunked_text[i] for i in top_k_indices]

    return results


# In[26]:


search_documents("What are prohibited ai practices?", top_k=2)


# ## Generation

# In[27]:


from transformers import pipeline

from genaibook.core import get_device

device = get_device()
generator = pipeline(
    "text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct", device=device
)


# In[28]:


def generate_answer(query):
    # Retrieve relevant chunks
    context_chunks = search_documents(query, top_k=2)

    # Combine the chunks into a single context string
    context = "\n".join(context_chunks)

    # Generate a response using the context
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    # Define the context to be passed to the model
    system_prompt = (
        "You are a friendly assistant that answers questions about the AI Act. "
        "If the user is not making a question, you can ask for clarification"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    response = generator(messages, max_new_tokens=300)
    return response[0]["generated_text"][2]["content"]


# In[29]:


answer = generate_answer("What are prohibited ai practices in the EU act?")
print(answer)


# In[ ]:




