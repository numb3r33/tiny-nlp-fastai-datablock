from fastai.text.all import *
import re

class MiniTokenizer(Transform):
    "A minimal whitespace tokenizer that lowercases and strips punctuation."
    def encodes(self, text:str):
        """
        Converts a raw text string into a list of cleaned tokens.

        This is our 'Instruction Set':
        1. LOWER: text.lower()
        2. CLEAN_PUNCT: re.sub(...)
        3. SPLIT: .split()

        >>> # This is a doctest. It will be run to verify the code.
        >>> tokenizer = MiniTokenizer()
        >>> tokenizer("Hello, World! This is a test.")
        ['hello', 'world', 'this', 'is', 'a', 'test']
        """
        # 1. LOWER
        text = text.lower()
        # 2. CLEAN_PUNCT: Remove all characters that are not letters, numbers, or whitespace
        text = re.sub(r'[^\w\s]', '', text)
        # 3. SPLIT
        return text.split()