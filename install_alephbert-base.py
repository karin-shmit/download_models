from transformers import BertModel, BertTokenizerFast
import os

# Specify the model name from Hugging Face
model_name = "onlplab/alephbert-base"

# Define the local directory to save the model
local_dir = f"./models/{model_name}"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download and save the tokenizer and model
alephbert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
alephbert = BertModel.from_pretrained(model_name)

# Save the model and tokenizer locally
alephbert_tokenizer.save_pretrained(local_dir)
alephbert.save_pretrained(local_dir)

print(f"Model and tokenizer saved to {local_dir}")
