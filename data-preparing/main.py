import re
from simple_tokenizer import SimpleTokenizerV1
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# split text into tokens
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_worlds = sorted(set(preprocessed))

vocab = {token:integer for integer, token in enumerate(all_worlds)}

tokenizer = SimpleTokenizerV1(vocab)
text = """
    "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
    """
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))
