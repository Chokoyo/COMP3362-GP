from typing import Any
import numpy as np
import numpy.typing as npt

class TokenDictInterface:
    def __init__(self) -> None:
        """
        Constructor for the Tokenizer object.
        """
        raise NotImplementedError

    def _count(self, token):
        """
        Count the number of times a token has been seen.
        """
        raise NotImplementedError

    def prune(self, min_count: int) -> None:
        """
        Prune the dictionary to remove tokens that have been seen less than min_count times.
        """
        raise NotImplementedError

    @property
    def dict(self):
        """
        Return the dictionary of tokens.
        """
        raise NotImplementedError

    @property
    def tokens(self):
        """
        Return the list of words in the vocab.
        """
        raise NotImplementedError


class TokenizerInterface:
    def __init__(self, vocab: dict) -> None:
        """
        Constructor for the Tokenizer object.
        Input: a dictionary of all legal tokens
        """
        raise NotImplementedError

    def tokenize(self, text: npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Tokenizes a list of LaTeX sequence into a list of symbols.
        Input: a numpy array of text string
        Output:a numpy array of tokenized text in Python list
        """
        raise NotImplementedError
        

class TokenDict(TokenDictInterface):
    def __init__(self):
        self.builtin = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self._tokens = {token: 0 for token in self.builtin} # dict of tokens and their counts
    
    def account(self, token_list):
        for token in token_list.split(' '):
            self._count(token)
            
    def _count(self, token):
        if token in self._tokens:
            self._tokens[token] += 1
        else:
            self._tokens[token] = 1
        return 1

    def prune(self, min_count: int = 24) -> None:
        # Remove artifacts
        if "Object]" in self._tokens: del self._tokens["Object]"]
        if "[object" in self._tokens: del self._tokens["[object"]
        self._tokens = {k: v for k, v in self._tokens.items() if v >= min_count or k in self.builtin}
        self._tokens = {k: v for v, k in enumerate(self._tokens.keys())}
    
    @property
    def dict(self):
        return {token: idx for idx, token in enumerate(self._tokens.keys())}
    
    @property
    def tokens(self):
        return self._tokens



class Tokenizer(TokenizerInterface):
    def __init__(self, vocab: dict):
        self.vocab = vocab

    def tokenize(self, text: npt.ArrayLike):
        """
        Tokenizes a list of LaTeX sequence into a list of symbols.
        Input: a numpy array of text string
        Output:a numpy array of tokenized text in Python list
        """
        tokenized_text = []
        for latex in text:
            tokenized_latex = []
            for symbol in latex.split(' '):
                if symbol in self.vocab:
                    tokenized_latex.append(symbol)
                else:
                    tokenized_latex.append("<UNK>")
            tokenized_text.append(tokenized_latex)
        return np.array(tokenized_text, dtype=object)

    def tokenize_idx(self, text: npt.ArrayLike):
        """
        Tokenizes a list of LaTeX sequence into a list of symbols.
        Input: a numpy array of text string
        Output:a numpy array of tokenized text in Python list
        """
        tokenized_text = []
        for latex in text:
            tokenized_latex = []
            for symbol in latex.split(' '):
                if symbol in self.vocab:
                    tokenized_latex.append(self.vocab[symbol])
                else:
                    tokenized_latex.append(self.vocab["<UNK>"])
            tokenized_text.append(tokenized_latex)
        return np.array(tokenized_text, dtype=object)
