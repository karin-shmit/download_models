from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Specify the model name from Hugging Face
model_name = "meta-llama/Llama-3.2-3B-Instruct"

# Define the local directory to save the model
local_dir = f"./models/{model_name}"

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    torch_dtype="auto",  # Automatically selects the appropriate dtype for the device
    device_map="auto"   # Automatically maps the model layers across available devices
)

# Put model in evaluation mode
model.eval()

# Example text input
input_text = "What is the future of AI?"

# Tokenize input and move tensors to the appropriate device
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

# Generate text
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text: ", generated_text)
