from transformers import BertModel, BertTokenizerFast
import torch
import os

# Specify the model name from Hugging Face
model_name = "onlplab/alephbert-base"


# Specify the local directory where the model is saved
local_dir = f"./models/{model_name}"

# Load the tokenizer and model from the local directory
alephbert_tokenizer = BertTokenizerFast.from_pretrained(local_dir)
alephbert = BertModel.from_pretrained(local_dir)

# Put model in evaluation mode
alephbert.eval()

# Example text input in Hebrew (AlephBERT is optimized for Hebrew text)
input_text = "שלום עולם"

# Tokenize input
input_ids = alephbert_tokenizer.encode(input_text, return_tensors='pt')

# Get hidden states from the model
with torch.no_grad():
    outputs = alephbert(input_ids)
    last_hidden_states = outputs.last_hidden_state

# Print the shape of the hidden states