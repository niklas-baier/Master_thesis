#!/usr/bin/env python
# coding: utf-8

# # Fine-Tuning Language Models
# 
# This notebook is a supplementary material for the Fine-Tuning Language Models of the [Hands-On Generative AI with Transformers and Diffusion Models book](https://learning.oreilly.com/library/view/hands-on-generative-ai/9781098149239/). This notebook includes:
# 
# * The code from the book
# * Additional examples
# * Exercise solutions

# ## Classifying Text
# 

# In[3]:


get_ipython().system('pip install datasets')
get_ipython().system('pip install peft')
get_ipython().system('pip install trl')
get_ipython().system('pip install numpy')
get_ipython().system('pip install torch')
get_ipython().system('pip install transformeres')
import datasets
import numpy as np
import torch
import transformers

np.set_printoptions(edgeitems=10, linewidth=70)
torch.set_printoptions(edgeitems=10, linewidth=70)


transformers.logging.set_verbosity_warning()
datasets.logging.set_verbosity_error()


# In[4]:


"""from huggingface_hub import snapshot_download
from tqdm import tqdm 
# Set the model ID for Phi-2

model_id = "microsoft/Phi-3.5-mini-instruct"
# Set the destination path
destination_path = "/media/niklas/SSD2/models/Phi3"

# Download the model
snapshot_download(
    repo_id=model_id,
    local_dir=destination_path,
    local_dir_use_symlinks=False,  # This ensures files are actually copied
    tqdm_class=tqdm
)

print(f"Model downloaded to {destination_path}")"""


# In[7]:


"""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "/media/niklas/SSD2/models/Phi3", 
    device_map="cuda", 
    torch_dtype=torch.bfloat16, 
   
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
"""


# In[2]:


"""from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model on CPU
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights

# Initialize an empty model and load it in shards
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        "/media/niklas/SSD2/models", 
        device_map="balanced"  # Balance model across devices
    )
tokenizer = AutoTokenizer.from_pretrained("/media/niklas/SSD2/models")"""


# In[3]:


"""def generate_text(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with memory-efficient settings
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        use_cache=True,        # Important for memory efficiency
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "Write a short story about"
print(generate_text(prompt))"""


# ### Identify a Dataset
# 

# In[4]:


from datasets import load_dataset

raw_datasets = load_dataset("ag_news")
raw_datasets


# In[5]:


raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]


# In[6]:


print(raw_train_dataset.features)


# ### Preprocess the Dataset

# In[7]:


from transformers import AutoTokenizer

checkpoint = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(batch):
    return tokenizer(
        batch["text"], truncation=True, padding=True, return_tensors="pt"
    )


tokenize_function(raw_train_dataset[:2])


# In[8]:


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets


# ### Define evaluation metrics
# 

# In[ ]:


get_ipython().system('pip install evaluate')
import evaluate

accuracy = evaluate.load("accuracy")
print(accuracy.description)
print(accuracy.compute(references=[0, 1, 0, 1], predictions=[1, 0, 0, 1]))


# In[ ]:


f1_score = evaluate.load("f1")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Compute accuracy and F1 Score
    acc_result = accuracy.compute(references=labels, predictions=preds)
    acc = acc_result["accuracy"]

    f1_result = f1_score.compute(
        references=labels, predictions=preds, average="weighted"
    )
    f1 = f1_result["f1"]

    return {"accuracy": acc, "f1": f1}


# ### Train the Model
# 

# In[ ]:


import torch
from transformers import AutoModelForSequenceClassification

from genaibook import get_device

device = get_device()
num_labels = 4
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=num_labels
).to(device)


# In[ ]:


from transformers import TrainingArguments

batch_size = 32  # You can change this if you have a big or small GPU
training_args = TrainingArguments(
    "classifier-chapter4",
    push_to_hub=True,
    num_train_epochs=2,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)


# In[ ]:


from transformers import Trainer

# Shuffle the dataset and pick 10,000 examples for training
shuffled_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_split = shuffled_dataset.select(range(10000))

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=small_split,
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.push_to_hub()


# In[ ]:


# Use a pipeline as a high-level help
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="genaibook/classifier-chapter4",
    device="cuda",
)
pipe(
    """The soccer match between Spain and 
Portugal ended in a terrible result for Portugal."""
)


# In[ ]:


# Get prediction for all samples
model_preds = pipe.predict(tokenized_datasets["test"]["text"])

# Get the dataset labels
references = tokenized_datasets["test"]["label"]

# Get the list of label names
label_names = raw_train_dataset.features["label"].names

# Print results of the first 3 samples
samples = 3
texts = tokenized_datasets["test"]["text"][:samples]
for pred, ref, text in zip(model_preds[:samples], references[:samples], texts):
    print(f"Predicted {pred['label']}; Actual {label_names[ref]};")
    print(text)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Convert predicted labels to ids
label_to_id = {name: i for i, name in enumerate(label_names)}
pred_labels = [label_to_id[pred["label"]] for pred in model_preds]

# Compute confusion matrix
confusion_matrix = evaluate.load("confusion_matrix")
cm = confusion_matrix.compute(
    references=references, predictions=pred_labels, normalize="true"
)["confusion_matrix"]

# Plot the confusion matrix
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
plt.title("Normalized confusion matrix")
plt.show()


# ## Generating text
# 

# In[ ]:


filtered_datasets = raw_datasets.filter(lambda example: example["label"] == 2)
filtered_datasets = filtered_datasets.remove_columns("label")


# ### Training a Generative Model

# In[ ]:


from transformers import AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = (
    tokenizer.eos_token
)  # Needed as SmolLM does not specify padding token.
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)


# In[ ]:


def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True)

tokenized_datasets = filtered_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],  # We only need the input_ids and attention_mask
)


# In[ ]:


tokenized_datasets


# In[ ]:


from transformers import DataCollatorForLanguageModeling

# mlm corresponds to masked language modeling
# and we set it to False as we are not training a masked language model
# but a causal language model
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# In[ ]:


samples = [tokenized_datasets["train"][i] for i in range(3)]

for sample in samples:
    print(f"input_ids shape: {len(sample['input_ids'])}")


# In[ ]:


out = data_collator(samples)
for key in out:
    print(f"{key} shape: {out[key].shape}")


# In[ ]:


training_args = TrainingArguments(
    "business-news-generator",
    push_to_hub=True,
    per_device_train_batch_size=8,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=200,
)


# In[ ]:


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"].select(range(5000)),
    eval_dataset=tokenized_datasets["test"],
)


# In[ ]:


trainer.train()


# In[ ]:


trainer.push_to_hub()


# In[ ]:


from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="genaibook/business-news-generator",
    device=device,
)
print(
    pipe("Q1", do_sample=True, temperature=0.1, max_new_tokens=30)[0][
        "generated_text"
    ]
)
print(
    pipe("Wall", do_sample=True, temperature=0.1, max_new_tokens=30)[0][
        "generated_text"
    ]
)
print(
    pipe("Google", do_sample=True, temperature=0.1, max_new_tokens=30)[0][
        "generated_text"
    ]
)


# ## A Quick Introduction to Adapters

# In[ ]:


from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()


# ## A light introduction to quantization

# In[ ]:


model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)


# In[ ]:


import numpy as np

def scaling_factor(vector):
    # Get largest value of vector
    m = np.max(np.abs(vector))

    # Return scaling factor
    return 127 / m


array = [1.2, -0.5, -4.3, 1.2, -3.1, 0.8, 2.4, 5.4, 0.3]
alpha = scaling_factor(array)
quantized_array = np.round(alpha * np.array(array)).astype(np.int8)
dequantized_array = quantized_array / alpha

print(f"Scaling factor: {alpha}")
print(f"Quantized array: {quantized_array}")
print(f"Dequantized array: {dequantized_array}")
print(f"Difference: {array - dequantized_array}")


# In[ ]:


from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", 
    quantization_config=quantization_config
)


# ## Putting It All Together
# 

# In[ ]:


quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=quantization_config,
    device_map="auto",
)


# In[ ]:


from trl import SFTConfig, SFTTrainer

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    "fine_tune_e2e",
    push_to_hub=True,
    per_device_train_batch_size=8,
    weight_decay=0.1,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=200,
    gradient_checkpointing=True,
    max_seq_length=512,
    # New parameters
    dataset_text_field="text",
    packing=True,
)

trainer = SFTTrainer(
    model,
    args=sft_config,
    train_dataset=dataset.select(range(300)),
    peft_config=peft_config,
)

trainer.train()


# In[ ]:


trainer.push_to_hub()


# In[ ]:


# We load the base model just as before
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    torch_dtype=torch.float16,
    device_map="auto",
)

# You can load the adapter with `load_adapter`
# Then load the model with `from_pretrained`.
model.load_adapter("genaibook/fine_tune_e2e")  # change with your adapter name

# Alternatively, you could just use `from_pretrained` with the adapter name and it
# will automatically take care of loading the base and adapter models.
# model = AutoModelForCausalLM.from_pretrained("genaibook/fine_tune_e2e"...

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
pipe("### Human: Hello!### Assistant:", max_new_tokens=100)


# In[ ]:


pipe = pipeline(
    "text-generation", "HuggingFaceTB/SmolLM-135M-Instruct", device=device
)
messages = [
    {
        "role": "system",
        "content": """You are a friendly chatbot who always responds 
        in the style of a pirate""",
    },
    {
        "role": "user",
        "content": "How many helicopters can a human eat in one sitting?",
    },
]
print(pipe(messages, max_new_tokens=128)[0]["generated_text"][-1])


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")

chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {
        "role": "assistant",
        "content": "I'm doing great. How can I help you today?",
    },
    {
        "role": "user",
        "content": "I'd like to show off how chat templating works!",
    },
]

tokenizer.apply_chat_template(chat, tokenize=False)


# In[ ]:


print(tokenizer.apply_chat_template(chat, tokenize=False))


# 

# ## Solutions
# 
# A big part of learning is putting your knowledge into practice. We strongly suggest not looking at the solutions before taking a stab at the problem. Scroll down for th answers

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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Exercises 

# **1. What's the difference between base and fine-tuned models? What kind of model is a conversational one?** 
# 
# * **Base models:** Base models are models that are trained on a large dataset to perform general language modeling tasks. They serve as the model that can be further adapted to specific tasks.
# * **Fine-tuned models:**  Fine-tuned models are derived from base models by training them further on a smaller, more specific dataset tailored for a particular task or domain. Fine-tuning adjusts the pretrained model's parameters to enhance its performance in specialized areas. Examples include models fine-tuned for summarization, text classification, named entity recognition, and more.
# * **Conversational models:** Conversational models are a type of fine-tuned model specifically trained to generate responses in a conversational context. These models are fine-tuned on datasets composed of dialogue or chat-like interactions, enabling them to produce coherent replies in a conversational setting.

# **2. In which cases would you pick a base encoder model for fine-tuning?**
# 
# Fine-tuning often makes sense when dealing with tasks that require understanding, such as classification, named entity recognition, and question answering. Encoder models, like BERT, usually shine due to their ability to produce contextual embeddings and their computational efficiency. BERT models are often used as base models for fine-tuning because they provide a strong foundation for a wide range of NLP tasks.
# 
# However, there are cases where a base encoder model might be sufficient without fine-tuning:
# 
# * **Strong Zero-shot or Few-shot capabilities** In such cases, the base model can be used directly to achieve good performance on specific tasks without additional training.
# * **Resource constraints** Fine-tuning requires additional computational resources and time.
# * **General tasks** For tasks that are either generic or too similar to the pre-training objectives, the base model might perform sufficiently well without fine-tuning.

# **3. Explain the differences between fine-tuning, instruct-tuning, and QLoRA.**
# 
# * **Fine-tuning:** Pick a pre-trained model and keep training to adapt/specialize it on a task or domain. The model's parameters are adjusted to improve its performance on the target task.
# * **Instruct-tuning:** A type of fine-tuning in which an instruct dataset is used. This dataset can formulate different tasks as instructions, helping the model generalize to solve new tasks.
# * **QLoRA:** LoRA is a type of fine-tuning in which just an adapter (an additional small set of parameters) is modified while the base model is frozen. QLoRA is a variant of LoRA that uses quantization to quantize the base model and hence require less GPU memory.

# **4. Does using adapters lead to a larger model size?**
# 
# Adapters initially add a very small overhead to the model size. Fortunately, we can merge back the adapter weights to the base model, going back to the original size. To achieve this, you can use `merge_and_unload`.

# **5. How much GPU memory is needed to load a 70B model in half-precision, 8-bit quantization, and 4-bit quantization?**
# 
# A quick (not extremely precise, but gives a good idea of order of magnitude):
# 
# * **Full Precision**: 32-bits (4 bytes) for each of the 70B params. 70B * 4 bytes = 280 GB
# * **Half Precision**: 16-bits (2 bytes) for each of the 70B params. 70B * 2 bytes = 140 GB
# * **8-bit Quantization**: 8-bits (1 byte) for each of the 70B params. 70B * 1 byte = 70 GB
# * **4-bit Quantization**: 4-bits (0.5 bytes) for each of the 70B params. 70B * 0.5 bytes = 35 GB
# 
# Note: there's usually an additional overhead of other things being loaded into the GPU.

# **6. Why does QLoRA lead to slower training?** 
# 
# QLoRA decreases the memory requirements but increases the computational overhead due to additional quantization and dequantization steps during training. These extra steps require additional computation, leading to slower training.

# **7. In which cases do we freeze the model weights during fine-tuning?**
# 
# In the context of transformers fine-tuning, the traditional process does not freeze the model weights. However, there are cases where freezing the model weights can be beneficial, such as PEFT or using the model as a feature extractor. Sometimes, you might want to use the pre-trained model as a feature extractor. In such cases, you freeze the model weights and use the model to extract features for downstream tasks. For example, you might want to connect a traditional model (e.g. SVM, Random Forest) to the transformer model's output. If you have very limited data, it might make sense to freeze most of the model and only fine-tune the last layers.

# ### Challenges
# 
# **8. Image Classification** Although this chapter has focused on fine-tuning transformer models for NLP tasks, transformers can also be used for other modalities such as audio and Computer Vision. The goal of this challenge is to fine-tune a transformer model for image classification. We suggest to:
# 
# * Use a pre-trained vision transformer model such as [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k).
# * Use a dataset of images and labels such as [food101](https://huggingface.co/datasets/food101).
# 

# The `food101` dataset contains 101 food categories, with a total of 101,000 images. We'll just use a small subset of the data (5,000 images) to validate that the fine-tuning process works. Feel free to use more data and run additional experiments!

# In[ ]:


from datasets import load_dataset

food = load_dataset("food101", split="train[:5000]")


# In[ ]:


food


# Let's create train and test splits out of this dataset. 

# In[ ]:


food = food.train_test_split(test_size=0.2, seed=42)


# In[ ]:


sample = food["train"][1]
sample 


# In[ ]:


sample["image"]


# That looks tasty. What is it? What does label 6 correspond to in the dataset?

# In[ ]:


labels = food["train"].features["label"].names
labels[6]


# Yummy! Note that the 5,000 samples in our subset of the dataset don't represent the 101 categories, but just a few of them. Let's see what we have in the train and test sets:

# In[ ]:


import numpy as np
np.unique(np.array(food["train"]["label"]), return_counts=True)


# In[ ]:


np.unique(np.array(food["test"]["label"]), return_counts=True)


# We only have 7 classes out of the 101 in the original dataset. We'll train our classifier so it learns to recognize photos among those 7 classes. To do so, we need to use class ids between 0 and 6, and not the original numbers we see above. This is because the classifier will select the chosen class based on the order of the output cell whose probability is larger, so numbers must be correlative and start at 0.
# 
# An alternative would be to pass the 101 classes as potential outputs but only use examples for the 7 classes in the subset. The model will still learn (you can verify it yourself!), but it will be slightly harder for it :)

# Let's then create dictionaries from the new labels to the class names and vice versa. We'll also create a mapping from the original label ids (all the classes in the dataset) to the new ones between 0 and 6 we are fine-tuning for.

# In[ ]:


label2id, id2label, original2id = dict(), dict(), dict()
finetuning_label_ids = np.unique(np.array(food["test"]["label"]))
for i, old_label_id in enumerate(finetuning_label_ids):
    label = labels[old_label_id]
    label2id[label] = str(i)
    id2label[str(i)] = label
    original2id[old_label_id] = i


# In[ ]:


id2label


# Let's now pre-process the dataset. We'll use an `AutoImageProcessor` rather than an `AutoTokenizer` to preprocess the images. Just as with tokenizers, the processor is tied to a model. This ensures we pass images in the right format to the model.

# In[ ]:


from transformers import AutoImageProcessor

model_id = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_id)


# If you take a look at the [preprocessor configuration](https://huggingface.co/google/vit-base-patch16-224-in21k/blob/main/preprocessor_config.json), you'll see that it resizes the input image to a size of 224x224 and normalizes so that input values are between -1 and 1. It can also take care of converting the input PIL images to PyTorch tensors:

# In[ ]:


preprocessed = image_processor(images=[sample["image"]], return_tensors="pt")["pixel_values"][0]
preprocessed.shape, preprocessed.min(), preprocessed.max()


# We'll use the built-in processor to make sure that input data follows the same transformations that were applied during pre-training. Alternatively, you could explore other transformations if they make sense for your task, such as cropping to square sizes before resizing (if keeping the aspect ratio is important for your task), or performing data augmentation.
# 
# Another step of the transformation is to map the original class ids to the new ones.

# In[ ]:


def preprocess(examples):
    examples.update(image_processor(examples["image"], return_tensors="pt"))
    examples["label"] = [original2id[x] for x in examples["label"]]
    del examples["image"]
    return examples


# In[ ]:


food = food.with_transform(preprocess)


# We'll also use the `DefaultDataCollator` to collate the data into batches. This collator does not apply any preprocessing - it just batches the data. It can be used in situations where, like in our case, all input tensors already have the same shape.

# In[ ]:


from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()


# We'll evaluate the model with the accuracy metric.

# In[ ]:


from transformers import Trainer, TrainingArguments
import evaluate


# In[ ]:


accuracy = evaluate.load("accuracy")


# In[ ]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Finally, let's load the base model. Although not strictly needed, we can add `id2label` and `label2id` to the model. This will help us convert between class names and labels and add this information to the model `config.json` file.

# In[ ]:


from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    model_id,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)


# Let's now train! Things are similar as before.

# In[ ]:


batch_size = 32 # change according to GPU

training_args = TrainingArguments(
    "my-food-model",
    num_train_epochs=3,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=5e-5,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True, 
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()


# We got a test accuracy of almost 94% in three epochs.

# In[ ]:


trainer.push_to_hub()


# Let's now use the `image-classification` pipeline and try to classify the image from the beginning.

# In[ ]:


from transformers import pipeline

classifier = pipeline("image-classification", model="pcuenq/my-food-model")
classifier(sample["image"])


# Nice! Feel free to explore:
# 
# * Find samples from the validation set for which the model is very confident but mismatches the label. Which one is right?
# * Try using other models, such as Swin Transformer, MobileViT or ConvNNeXT.
# * Explore creating your own dataset with just 3 clases and 10 images per class. How does the model perform?
