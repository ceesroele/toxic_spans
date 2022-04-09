import csv
import json
import ast
from spacy.tokens import DocBin, Span
import spacy
from collections import Counter
from util import read_data, make_spans


SILENT = True


def create_docbin(fname: str, basename: str, nlp, spans_key='sc'):
    """Create a DocBin from a dataframe with data in columns ['spans', 'text']"""
    doc_bin = DocBin()
    errors = 0
    texts = 0
    spans_n = 0
    len_spans = Counter()
    for spans, text in read_data(fname):
        texts += 1
        ms = make_spans(spans)
        doc = nlp(text)
        span_lst = []
        for start, end in ms:
            spans_n += 1
            span = doc.char_span(start, end, label='TOXIC')
            if span is None:
                errors += 1
            else:
                len_spans[len(span)] += 1
                span_lst.append(span)
        doc.set_ents(span_lst)
        doc.spans[spans_key] = list(doc.ents)
        doc_bin.add(doc)
    doc_bin.to_disk(f'corpus/{basename}.spacy')
    if not SILENT:
        print(f"Processing errors {basename}: ", errors)
        print(f"Texts: {texts}. Spans: {spans_n}")
        print(len_spans)


def main():
    nlp = spacy.load("en_core_web_sm")

    create_docbin('assets/tsd_train.csv', 'train', nlp, spans_key='txs')
    create_docbin('assets/tsd_trial.csv', 'dev', nlp, spans_key='txs')
    create_docbin('assets/tsd_test.csv', 'eval', nlp, spans_key='txs')


if __name__ == '__main__':
    main()
