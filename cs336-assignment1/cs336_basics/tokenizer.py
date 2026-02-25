from typing import TypeAlias
import regex as re

WordFreqs: TypeAlias = dict[tuple[bytes, ...], int]


def pretokenize(text: str, special_tokens: list[str]) -> WordFreqs:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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
