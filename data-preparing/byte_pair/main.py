import torch
import tiktoken
from data_loader import create_dataloader_v1

tokenizer = tiktoken.get_encoding("gpt2")

with open("../simple_tokenizer/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_lenght=4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_lenght=max_lenght, stride=max_lenght, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# print(f"tokens ids:\n{inputs}")
# print(f"\ninputs shape:\n{inputs.shape}")

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embeddings = token_embedding_layer(inputs)

context_length = max_lenght
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

# print(f"token embeddings shape:\n{token_embeddings.shape}\n")
# print(f"pos embeddings shape:\n{pos_embeddings.shape}\n")

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
