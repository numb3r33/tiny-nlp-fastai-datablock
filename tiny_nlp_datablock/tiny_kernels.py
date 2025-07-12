from fastai.text.all import *

class AddBigrams(Transform):
    "An 'additive' kernel that adds bigram features to a token list."
    def encodes(self, tokens:list):
        """
        Adds n-gram of order 2 (bigrams) to the end of the token list.

        >>> #Before: A  simple list of 6 tokens
        >>> tokens_in = ['hello', 'world', 'this', 'is', 'a', 'test']
        >>> kernel = AddBigrams()
        >>> #After: Original tokens + 5 new bigram tokens
        >>> kernel(tokens_in)
        ['hello', 'world', 'this', 'is', 'a', 'test', 'hello_world', 'world_this', 'this_is', 'is_a', 'a_test']
        """

        if len(tokens) < 2: return tokens
        # Create bigrams like 'hello_world', 'world_this', etc.
        bigrams = ['_'.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        return tokens + bigrams
    
class DuplicateRareTokens(Transform):
    "A 'multiplicative' kernel that duplicates rare tokens to increase the signal."

    def __init__(self, rare_words:list=None, n_copies:int=2):
        self.rare_words = set(rare_words) if rare_words else set()
        self.n_copies = n_copies
    
    def encodes(self, tokens:list):
        """
        Duplicates any token found in `rare_words` list.

        >>> # Define 'abysmal' as a rare word we care about
        >>> rare = ['abysmal', 'magnificent']
        >>> kernel = DuplicateRareTokens(rare_words=rare, n_copies=2)
        >>> # Before: 'abysmal' appears once
        >>> tokens_in = ['this', 'movie', 'was', 'truly', 'abysmal']
        >>> # After: 'abysmal' appears 1 (original) + 2 (copies) = 3 times
        >>> kernel(tokens_in)
        ['this', 'movie', 'was', 'truly', 'abysmal', 'abysmal', 'abysmal']
        """

        new_tokens = []
        for t in tokens:
            new_tokens.append(t)
            if t in self.rare_words:
                # Add n_copies more
                new_tokens.extend([t] * (self.n_copies))
        return new_tokens
