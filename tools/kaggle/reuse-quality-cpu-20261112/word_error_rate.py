"""Transparent transcript agreement; the caller chooses the reference's authority.

Default normalization uses Unicode NFKC, case-folding and letter/digit runs.
All punctuation (including apostrophes, hyphens and underscores) separates words.
No stemming, contraction expansion, number rewriting or ASR-error correction.
"""
import argparse
import json
from pathlib import Path
import tempfile
import unicodedata
from typing import Sequence


def normalize_words(text: str) -> list[str]:
    if not isinstance(text,str):
        raise TypeError('transcript must be text')
    normalized = unicodedata.normalize('NFKC',text).casefold()
    return ''.join(char if char.isalnum() else ' ' for char in normalized).split()


def word_errors(reference: Sequence[str], hypothesis: Sequence[str]) -> dict:
    """Unit-cost Levenshtein alignment; ties prefer substitution, deletion, insertion.

    WER is null for an empty reference; counts still describe all insertions.
    Token indices in the alignment are zero-based; absent tokens have null indices.
    """
    if isinstance(reference,str) or isinstance(hypothesis,str):
        raise TypeError('word_errors expects token sequences; use compare_transcripts for text')
    ref, hyp = list(reference), list(hypothesis)
    if any(not isinstance(token,str) or not token for token in ref+hyp):
        raise ValueError('tokens must be nonempty strings')
    distance = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1):distance[i][0] = i
    for j in range(len(hyp)+1):distance[0][j] = j
    for i in range(1,len(ref)+1):
        for j in range(1,len(hyp)+1):
            distance[i][j] = min(distance[i-1][j-1]+(ref[i-1]!=hyp[j-1]),
                                 distance[i-1][j]+1,distance[i][j-1]+1)
    i, j = len(ref), len(hyp)
    alignment = []
    counts = {'correct':0,'substitutions':0,'deletions':0,'insertions':0}
    while i or j:
        if i and j and ref[i-1] == hyp[j-1] and distance[i][j] == distance[i-1][j-1]:
            operation, ri, hi = 'equal', i-1, j-1
            counts['correct'] += 1
            i, j = i-1, j-1
        elif i and j and distance[i][j] == distance[i-1][j-1]+1:
            operation, ri, hi = 'substitute', i-1, j-1
            counts['substitutions'] += 1
            i, j = i-1, j-1
        elif i and distance[i][j] == distance[i-1][j]+1:
            operation, ri, hi = 'delete', i-1, None
            counts['deletions'] += 1
            i -= 1
        elif j and distance[i][j] == distance[i][j-1]+1:
            operation, ri, hi = 'insert', None, j-1
            counts['insertions'] += 1
            j -= 1
        else:
            raise RuntimeError('inconsistent edit-distance alignment')
        alignment.append({'operation':operation,'reference_index':ri,'hypothesis_index':hi,
                          'reference':None if ri is None else ref[ri],
                          'hypothesis':None if hi is None else hyp[hi]})
    alignment.reverse()
    edits = counts['substitutions']+counts['deletions']+counts['insertions']
    return {**counts,'reference_words':len(ref),'hypothesis_words':len(hyp),'edits':edits,
            'wer':edits/len(ref) if ref else None,'alignment':alignment}


def compare_transcripts(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    return {'normalized_reference':ref,'normalized_hypothesis':hyp,**word_errors(ref,hyp)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--hypothesis',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    result = compare_transcripts(args.reference.read_text(),args.hypothesis.read_text())
    text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',dir=args.output.parent,delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(text)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(text,end='')


if __name__ == '__main__':
    main()
