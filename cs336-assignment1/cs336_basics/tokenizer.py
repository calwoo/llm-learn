from collections.abc import Iterable, Iterator
from typing import TypeAlias
import regex as re
import os

WordFreqs: TypeAlias = dict[tuple[bytes, ...], int]
Vocab: TypeAlias = dict[int, bytes]


# GPT-2 pretokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, special_tokens: list[str]) -> WordFreqs:
    word_freqs: WordFreqs = {}

    # there should always be some special tokens, such as <|endoftext|>
    assert len(special_tokens) > 0
    special_split_pattern = "|".join(map(re.escape, special_tokens))
    chunks: list[str] = re.split(special_split_pattern, text)
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            match_grp = match.group()
            match_byteseq = match_grp.encode("utf-8")
            match_tokens = tuple(bytes([b]) for b in match_byteseq)

            if match_tokens not in word_freqs:
                word_freqs[match_tokens] = 0
            word_freqs[match_tokens] += 1

    return word_freqs


def apply_merge(word_freqs: WordFreqs, pair_to_merge: tuple[bytes, bytes]) -> WordFreqs:
    token_A, token_B = pair_to_merge
    new_token = token_A + token_B
    new_word_freqs = {}

    for word, freq in word_freqs.items():
        new_word = []
        skip_next = False
        for wtoken_A, wtoken_B in zip(word, word[1:]):
            if skip_next:
                skip_next = False
                continue
            if wtoken_A == token_A and wtoken_B == token_B:
                new_word.append(new_token)
                skip_next = True
            else:
                new_word.append(wtoken_A)

        if not skip_next:
            # don't forget the last token!
            new_word.append(word[-1])

        new_word = tuple(new_word)
        new_word_freqs[new_word] = freq

    return new_word_freqs


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[Vocab, list[tuple[bytes, bytes]]]:
    # initial vocab with individual bytes + special tokens
    vocab: Vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # read text and pretokenize
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    word_freqs: WordFreqs = pretokenize(text, special_tokens)

    # run BPE merging
    merges = []
    while len(vocab) < vocab_size:
        pair_counts = {}
        for word, freq in word_freqs.items():
            for wtoken_A, wtoken_B in zip(word, word[1:]):
                pair = (wtoken_A, wtoken_B)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq

        if len(pair_counts) == 0:
            break

        # get the highest count and merge
        pair_to_merge = max(pair_counts, key=lambda p: (pair_counts[p], p))
        word_freqs = apply_merge(word_freqs, pair_to_merge)

        # update vocab
        wtoken_A, wtoken_B = pair_to_merge
        new_token = wtoken_A + wtoken_B
        vocab[len(vocab)] = new_token
        merges.append(pair_to_merge)

    return vocab, merges


class Tokenizer:
    def __init__(self, vocab: Vocab, merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = None
        if special_tokens:
            self.special_tokens = list(sorted(special_tokens, key=len, reverse=True))

        self.inverse_vocab: dict[bytes, int] = {}
        for idx, token in vocab.items():
            self.inverse_vocab[token] = idx

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer": ...

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            chunks = [text]

            if self.special_tokens is not None:
                special_split_pattern = "|".join(map(re.escape, self.special_tokens))
                chunks = re.split(rf"({special_split_pattern})", text)

            for chunk in chunks:
                if self.special_tokens and chunk in self.special_tokens:
                    yield self.inverse_vocab[chunk.encode("utf-8")]
                    continue

                # if not special token, pretokenize and apply merges
                for match in re.finditer(PAT, chunk):
                    match_byteseq = match.group().encode("utf-8")
                    match_tokens = tuple(bytes([b]) for b in match_byteseq)
                    merged_tokens = self._apply_merges(list(match_tokens))
                    for tok in merged_tokens:
                        yield self.inverse_vocab[tok]

    def decode(self, ids: list[int]) -> str:
        decoded_bytes = [self.vocab[idx] for idx in ids]
        return b"".join(decoded_bytes).decode("utf-8", errors="replace")

    def _apply_merges(self, token_seq: list[bytes]) -> list[bytes]:
        for merge_A, merge_B in self.merges:
            if len(token_seq) == 1:
                return token_seq

            new_token_seq = []
            skip_next = False
            for token_A, token_B in zip(token_seq, token_seq[1:]):
                if skip_next:
                    skip_next = False
                    continue

                if token_A == merge_A and token_B == merge_B:
                    new_token = merge_A + merge_B
                    new_token_seq.append(new_token)
                    skip_next = True
                else:
                    new_token_seq.append(token_A)

            # add in last token
            if not skip_next:
                new_token_seq.append(token_seq[-1])

            token_seq = new_token_seq

        return token_seq
