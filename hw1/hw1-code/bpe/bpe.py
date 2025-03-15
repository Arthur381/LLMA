from collections import defaultdict, Counter

class Tokenizer:
    def __init__(self):
        self.merges = []        # List of merge operations (tuples) in the order they were learned.
        self.token2id = {}      # Mapping from token (str) to id (int)
        self.id2token = {}      # Reverse mapping from id to token
        self.vocab = {}         # Final vocabulary (token to id mapping)

    def train(self, text, vocab_size):
        """
        Train the tokenizer using the BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.
        Return:
            None
        """
        # 1. Tokenize the text into words.
        words = text.split()
        # Represent each word as a tuple of characters plus a special end-of-word marker.
        word_freqs = Counter()
        for word in words:
            # For example, "hello" becomes ('h', 'e', 'l', 'l', 'o', '</w>')
            tokens = list(word) + ['</w>']
            word_freqs[tuple(tokens)] += 1

        # 2. Initialize vocabulary with all individual characters and the marker.
        vocab = set()
        for word in word_freqs:
            vocab.update(word)

        # 3. Iteratively perform merge operations until we hit the desired vocab size.
        while len(vocab) < vocab_size:
            # Count frequency of adjacent symbol pairs in the words.
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pairs[pair] += freq

            if not pairs:
                break

            # Find the most frequent pair.
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # Merge the pair into a new token.
            new_token = ''.join(best_pair)
            vocab.add(new_token)

            # Update each word by replacing the best pair with the new token.
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word):
                    # If the pair is found, merge it.
                    if i < len(word) - 1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        # 4. Build token-to-id and id-to-token mappings (sorting tokens for consistency).
        sorted_tokens = sorted(vocab)
        self.token2id = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.vocab = self.token2id

    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.
        Return:
            ids (list): list of integer-type tokens.
        """
        token_ids = []
        # Process each word individually.
        for word in text.split():
            # Start by representing the word as a list of characters with an end marker.
            tokens = list(word) + ['</w>']
            # Apply each learned merge operation in the order they were learned.
            for merge in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    # If the merge pair is found, merge them.
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge:
                        new_tokens.append(''.join(merge))
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            # Convert the resulting tokens into ids.
            for token in tokens:
                token_id = self.token2id.get(token)
                # Fallback: if a token wasn't found (rare), break it into characters.
                if token_id is None:
                    for char in token:
                        token_ids.append(self.token2id.get(char))
                    continue
                token_ids.append(token_id)
        return token_ids

    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.
        Return:
            text (str): string-type data.
        """
        # Convert ids back to tokens.
        tokens = [self.id2token.get(i, '') for i in ids]
        words = []
        current_word = ""
        # Reconstruct the string by checking for the end-of-word marker.
        for token in tokens:
            if token.endswith('</w>'):
                # Remove the marker and complete the word.
                current_word += token[:-4]
                words.append(current_word)
                current_word = ""
            else:
                current_word += token
        if current_word:
            words.append(current_word)
        return " ".join(words)


# Example usage:
if __name__ == '__main__':
    sample_text = "this is a test test"
    tokenizer = Tokenizer()
    # Train with a target vocabulary size (e.g., 50 tokens)
    tokenizer.train(sample_text, vocab_size=50)
    
    encoded = tokenizer.encode(sample_text)
    print("Encoded token IDs:", encoded)
    
    decoded = tokenizer.decode(encoded)
    print("Decoded text:", decoded)
