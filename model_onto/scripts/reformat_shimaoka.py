import os

def reformat_file(inp_fp, out_fp, ontology_pieces=None):
    """
    Read the Shimaoka train/dev/test set and reformat it to
    the format that hiertype expects. 

    The Shimaoka data is based on OntoNotes and Figer downloaded here: 
    http://www.cl.ecei.tohoku.ac.jp/~shimaoka/corpus.zip
    I believe this dataset comes from this paper:
    https://arxiv.org/pdf/1606.01341.pdf
    The original OntoNotes data comes from here (needs sign up for license):
    https://catalog.ldc.upenn.edu/LDC2013T19
    The original Figer data (called Wiki in corpus.zip) comes from here:
    https://github.com/xiaoling/figer
    """
    if ontology_pieces is None:
        ontology_pieces = set()
    with open(inp_fp, 'r', encoding='utf-8') as infile, \
         open(out_fp, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                start, end, sent, type_list, rest = line.split('\t', maxsplit=4)
            except ValueError:
                print(f"Unable to parse: {line}")
                continue
            result = f"{sent}\t{start}:{end}\t{type_list}"
            types = set(type_list.split())
            ontology_pieces = ontology_pieces.union(types)
            outfile.write(result + os.linesep)
    return ontology_pieces

def reformat(indir, outdir):
    print(f"Reformatting {indir} -> {outdir}")
    ontology_fp = os.path.join(outdir, "ontology.txt")
    ontology_pieces = set()
    for root, dirs, files in os.walk(indir):
        os.makedirs(os.path.join(outdir, root), exist_ok=True)
        for fp in files:
            inp_fp = os.path.join(root, fp)
            out_fp = os.path.join(outdir, fp)
            print(f'Working on {inp_fp} -> {out_fp}')
            ontology_pieces = reformat_file(inp_fp, out_fp, ontology_pieces)
    ontology = sorted(list(ontology_pieces))
    with open(ontology_fp, 'w', encoding='utf-8') as outfile:
        for i in ontology:
            outfile.write(i + os.linesep)

if __name__ == '__main__':
    import fire
    fire.Fire(reformat)
