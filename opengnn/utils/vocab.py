from typing import Iterable, List, Optional, Union, Callable

import json

import tensorflow as tf

from opengnn.utils.misc import read_file


class Vocab:
    """Vocabulary class."""

    def __init__(self, special_tokens: Optional[Iterable[str]]=None):
        """Initializes a vocabulary.
        Args:
            special_tokens: A list of special tokens (e.g. start of sentence).
        """
        self._token_to_id = {}  # type: Dict[str, int]
        self._id_to_token = []  # type: List[str]
        self._frequency = []  # type: List[int]

        if special_tokens is not None:
            for index, token in enumerate(special_tokens):
                self._token_to_id[token] = index
                self._id_to_token.insert(index, token)

                # Set a very high frequency to avoid special tokens to be
                # pruned. Note that Python sort functions are stable which
                # means that special tokens in pruned vocabularies will have
                # the same index.
                self._frequency.insert(index, float("inf"))

    @property
    def size(self) -> int:
        """Returns the number of entries of the vocabulary."""
        return len(self._id_to_token)

    def add_from_file(self,
                      filename: str,
                      field: Optional[str]=None,
                      index: Optional[int]=None,
                      subtokenizer: Callable=None,
                      case_sentitive: bool=False) -> None:
        """Fills the vocabulary from a text file with json objects
        Args:
            list:
        """
        with read_file(filename) as f:
            for line in f:
                sequence = json.loads(line)
                if field is not None:
                    sequence = sequence[field]

                for token in sequence:
                    if index is not None:
                        token = token[index]

                    if subtokenizer is not None:
                        for subtoken in subtokenizer(token):
                            self.add(subtoken if case_sentitive else subtoken.lower())
                    else:
                        self.add(token if case_sentitive else token.lower())

    def serialize(self, path: str) -> None:
        """Writes the vocabulary on disk.
        Args:
            path: The path where the vocabulary will be saved.
        """
        with open(path, "wb") as vocab:
            for token in self._id_to_token:
                vocab.write(tf.compat.as_bytes(token))
                vocab.write(b"\n")


    def add(self, token: str) -> None:
        """Adds a token or increases its frequency.
        Args:
            token: The string to add.
        """
        if token not in self._token_to_id:
            index = self.size
            self._token_to_id[token] = index
            self._id_to_token.append(token)
            self._frequency.append(1)
        else:
            self._frequency[self._token_to_id[token]] += 1

    def lookup(self, identifier: Union[str, int], default: Optional[Union[str, int]]=None) -> Union[str, int]:
        """Lookups in the vocabulary.
        Args:
            identifier: A string or an index to lookup.
            default: The value to return if :obj:`identifier` is not found.
        Returns:
            The value associated with :obj:`identifier` or :obj:`default`.
        """
        value = None

        if isinstance(identifier, str):
            if identifier in self._token_to_id:
                value = self._token_to_id[identifier]
        elif identifier < self.size:
            value = self._id_to_token[identifier]

        if value is None:
            return default
        else:
            return value

    def prune(self, max_size: int=0, min_frequency: int=1) -> 'Vocab':
        """Creates a pruned version of the vocabulary.
        Args:
            max_size: The maximum vocabulary size.
            min_frequency: The minimum frequency of each entry.
        Returns:
            A new vocabulary.
        """
        sorted_ids = sorted(
            range(self.size), key=lambda k: self._frequency[k], reverse=True)
        new_size = len(sorted_ids)

        # Discard words that do not meet frequency requirements.
        for i in range(new_size - 1, 0, -1):
            index = sorted_ids[i]
            if self._frequency[index] < min_frequency:
                new_size -= 1
            else:
                break

        # Limit absolute size.
        if max_size > 0:
            new_size = min(new_size, max_size)

        new_vocab = Vocab()

        for i in range(new_size):
            index = sorted_ids[i]
            token = self._id_to_token[index]
            frequency = self._frequency[index]

            new_vocab._token_to_id[token] = i
            new_vocab._id_to_token.append(token)
            new_vocab._frequency.append(frequency)

        return new_vocab
