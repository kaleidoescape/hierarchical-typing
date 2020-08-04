from typing import *
import subprocess
import torch
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import numpy as np
from fire import Fire
import sys
from colors import blue
import h5py

from hiertype.contextualizers import Contextualizer, get_contextualizer
from hiertype.util.logging import setup_logger

logger = setup_logger()

T = TypeVar('T', covariant=True)


def get_file_length(corpus):
    #NOTE: use awk instead of wc -l, because wc -l only counts \n literal
    output = subprocess.check_output(["awk", "END{print NR}", corpus])
    sent_count = int(output.split()[0])
    return sent_count

def get_spans(lines: Iterator[str]) -> Iterator[Tuple[List[str], int, int]]:
    for line in lines:
        sentence, span, rest = line.split('\t')
        tokens = sentence.split(' ')
        l_str, r_str = span.split(':')
        l = int(l_str)
        r = int(r_str)
        yield (tokens, l, r, line)

def batched(xs: Iterator[T], batch_size: int = 128) -> Iterator[List[T]]:
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf

def select_embeddings(
        encoded: torch.Tensor,  # R[Batch, Layer, Word, Emb]
        mappings: List[List[int]],
        layers: List[int],
        unit: str = "subword"
) -> Iterator[torch.Tensor]:  # R[Layer, Word, Emb]

    bsz = encoded.size(0)

    for i in range(bsz):
        x = encoded[i, layers, :, :]  # R [Layer, Word, Emb]
        mapping = mappings[i]
        word_l = min(mapping)
        word_r = max(mapping) + 1
        subword_l = bisect_left(mapping, word_l)
        subword_r = bisect_right(mapping, word_r - 1)
        if subword_r == subword_l:
            subword_r += 1

        if unit == "subword":
            yield x[:, subword_l:subword_r, :]
        elif unit == "word":
            yield torch.cat(
                [
                    x[:, bisect_left(mapping, j):bisect_right(mapping, j), :].mean(dim=1, keepdim=True)
                    for j in range(word_l, word_r)
                ],
                dim=1
            )
        else:
            raise AssertionError("`unit` must be either `word` or `subword`")

def main(*,
         input_fp: str,
         output: str,
         model: str = "elmo-original",
         unit: str = "subword",
         batch_size: int = 64,
         layers: List[int],
         gpuid: int = 0,
         ):

    for k, v in reversed(list(locals().items())):  # seems that `locals()` stores the args in reverse order
        logger.info(f"{blue('--' + k)} \"{v}\"")
    
    file_length = get_file_length(input_fp)

    if gpuid >= 0:
        torch.cuda.set_device(gpuid)

    contextualizer: Contextualizer = get_contextualizer(
        model,
        device="cpu" if gpuid < 0 else f"cuda:{gpuid}",
        tokenizer_only=False
    )
    with h5py.File(output, 'w') as f, \
         open(input_fp, 'r') as inp:

        lines = tqdm(inp)

        spans: Iterator[Tuple[List[str], int, int]] = get_spans(lines)

        i = 0
        resized = False
        for batch in batched(spans, batch_size=batch_size):
            sentences, ls, rs, orig_lines = zip(*batch)

            tokenized_sentences, mappings = zip(*[
                contextualizer.tokenize_with_mapping(sentence)
                for sentence in sentences
            ])
            encoded = contextualizer.encode(tokenized_sentences, frozen=True)

            for j, emb in enumerate(select_embeddings(encoded, mappings, layers, unit)):
                x: np.ndarray = emb.detach().cpu().numpy()
                x_data = x.astype(np.float32)

                dset = f.create_dataset(str(i), data=x_data)
                dset.attrs['str'] = orig_lines[j]

                i += 1

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    Fire(main)
