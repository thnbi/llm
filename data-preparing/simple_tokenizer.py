import re
class SimpleTokenizerV1:
    def __init__(self, vocab):
        # Stores the vocabulary as as class attribute for acess in the encode and decode methods
        self.str_to_int = vocab
        # Creates an inverse vocabulary that maps token IDs back to the original text tokens
        self.int_to_str = {i:s for s, i in vocab.items()}

    # Processes input text into token IDs
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
                item.strip() for item in preprocessed if item.strip()
            ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # Convert token IDs back into text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Remove spaces before the specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
