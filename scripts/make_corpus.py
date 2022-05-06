import sys
from spacy.tokens import DocBin, Span
import spacy
from collections import Counter

sys.path.insert(0, './scripts')
from util import read_data
from evaluate_char import make_spans

SILENT = True

def test(nlp, spans, text, spans_key='sc'):
    ms = make_spans(spans)
    doc = nlp(text)
    span_lst = []
    len_spans = Counter()
    print(ms, text)
    for start, end in ms:
        span = doc.char_span(start, end, label='TOXIC')
        if span is None:
            print("broken for ", start, end, text[start:end])
        else:
            len_spans[len(span)] += 1
            span_lst.append(span)
    doc.set_ents(span_lst)
    doc.spans[spans_key] = list(doc.ents)


def find_errors(fname: str, basename: str, nlp, spans_key='sc', silent=SILENT):
    """Create a DocBin from a dataframe with data in columns ['spans', 'text']"""
    errors = 0
    texts = 0
    spans_n = 0
    len_spans = Counter()
    error_sentences = []
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
                error_sentences.append((spans, text))
            else:
                len_spans[len(span)] += 1
                span_lst.append(span)
        doc.set_ents(span_lst)
        doc.spans[spans_key] = list(doc.ents)
    if not silent:
        for e in error_sentences:
            print(e)
        print(f"Processing errors {basename}: ", errors)
        print(f"Texts: {texts}. Spans: {spans_n}")
        print(len_spans)
    return len_spans


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
        print(f"Texts: {texts}. Spans: {spans_n}")
        print(len_spans)
        print(f"Processing errors {basename}: ", errors)


def main():
    nlp = spacy.load("en_core_web_sm")

    create_docbin('assets/tsd_train.csv', 'train', nlp, spans_key='txs')
    create_docbin('assets/tsd_trial.csv', 'dev', nlp, spans_key='txs')
    create_docbin('assets/tsd_test.csv', 'eval', nlp, spans_key='txs')


if __name__ == '__main__':
    mode = 'make_corpus'
    #mode = 'find_errors'
    #mode = 'display_error'
    nlp = spacy.load('en_core_web_sm')

    if mode == 'make_corpus':
        main()
    elif mode == 'find_errors':
        find_errors('assets/tsd_train.csv', 'train', nlp, spans_key='txs', silent=False)
    elif mode == 'display_error':
        spans = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 40, 41, 42]
        text = 'Not a man.  A mouse trying hard to be a rat like Trump.'

        spans = [12, 13, 14, 15, 16, 17]
        text = 'so, are you morons happy about his visit?'

        spans = [1, 2, 3, 4, 5, 6, 7]
        text = "\"\"Garbage WE now pay for\"\". How much state tax did you contribute in the past three years?"
        test(nlp, spans, text)