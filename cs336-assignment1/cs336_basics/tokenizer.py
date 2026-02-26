from typing import TypeAlias
import regex as re

WordFreqs: TypeAlias = dict[tuple[bytes, ...], int]
Vocab: TypeAlias = dict[int, bytes]


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


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[Vocab, list[tuple[bytes, bytes]]]:
    # initial vocab with individual bytes + special tokens
    vocab: Vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # read text and pretokenize
    with open(input_path, "r", encoding="utf-8") as f:
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
        pair_to_merge = max(pair_counts, key=pair_counts.get)
        word_freqs = apply_merge(word_freqs, pair_to_merge)

        # update vocab
        wtoken_A, wtoken_B = pair_to_merge
        new_token = wtoken_A + wtoken_B
        vocab[len(vocab)] = new_token
        merges.append(pair_to_merge)

    return vocab, merges
