import re
from simple_tokenizer import SimpleTokenizerV1
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# split text into tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_worlds = sorted(set(preprocessed))
all_tokens = sorted(list(set(preprocessed)))

all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
tokenizer = SimpleTokenizerV1(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = "<|endoftext|> ".join((text1, text2))

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))
