from typing import TypeAlias

WordFreqs: TypeAlias = dict[tuple[bytes, ...], int]


def apply_merge(word_freqs: WordFreqs, pair_to_merge: tuple[bytes, bytes]) -> WordFreqs:
    token_A, token_B = pair_to_merge
    new_token = token_A + token_B
    new_word_freqs = {}

    for word, freq in word_freqs.items():
        new_word = []
        skip_next = False
        for wtoken_A, wtoken_B in zip(word, word[1:]):
            if skip_next:
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
        new_word_freqs[new_word] = freqs

    return new_word_freqs
