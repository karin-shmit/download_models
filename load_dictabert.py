from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

# Specify the model name from Hugging Face
model_name = "dicta-il/dictabert"

# Specify the local directory where the model is saved
local_dir = f"./models/{model_name}"

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForMaskedLM.from_pretrained(local_dir)

# Put model in evaluation mode
model.eval()

# Example text input with a masked token in Hebrew (assuming DictaBERT is optimized for Hebrew text)
input_text = "שלום [MASK] עולם"

# Tokenize input
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get model predictions
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# Get the predicted token for the masked position
masked_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
predicted_token_id = predictions[0, masked_index].argmax(dim=-1).item()
predicted_token = tokenizer.decode([predicted_token_id])

# Print the predicted token
print("Predicted Token for [MASK]: ", predicted_token)