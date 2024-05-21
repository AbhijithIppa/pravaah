import torch
from sentence_transformers import SentenceTransformer, util

# Check if a GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the question, answer, and responder_input
question = "Q: What is the difference between var, let, and const in JavaScript?"
answer = "A: Var has function-level scope, let has block-level scope, and const is a constant variable that cannot be reassigned."
responder_input = "Var: function-level scope, let: block-level scope, const: constant variable."

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)

# Get embeddings for the answer and responder_input
answer_embedding = model.encode(answer, convert_to_tensor=True).to(device)
responder_input_embedding = model.encode(responder_input, convert_to_tensor=True).to(device)


# Compute cosine similarity between the answer and responder_input
similarity_score = util.pytorch_cos_sim(answer_embedding, responder_input_embedding).item()

# Convert similarity score to a 5-mark scale
marks_out_of_5 = similarity_score * 5

print(f"Similarity score out of 5: {marks_out_of_5:.2f}")
