from ..util import read_data_csv
from ..evaluate_char import span2char, make_spans

import spacy


def test_span2char():
    """span2char: Check if spaCy spans correctly convert into character indexes"""
    nlp = spacy.load('en_core_web_sm')

    s = 'You behaved like an idiot office clerk.'

    doc = nlp(s)
    span = doc[4:6]

    assert span.text == 'idiot office'
    start_end_tuple = span2char(doc, span)
    assert start_end_tuple == (20, 32)
    assert s[start_end_tuple[0]:start_end_tuple[1]] == 'idiot office'


def test_span2char_csv():
    """span2char: test with data: Convert character spans to spaCy spans and back"""
    nlp = spacy.load('en_core_web_sm')
    max = 50
    counter = 0
    countspans = 0
    spanstrings = []
    for spans, text in read_data_csv('assets/tsd_train.csv'):
        doc = nlp(text)
        ms = make_spans(spans)
        for start, end in ms:
            span = doc.char_span(start, end, label='TOXIC')
            if span is None:
                print("broken for ", start, end, text[start:end])
            else:
                pred_start, pred_end = span2char(doc, span)
                assert pred_start == start
                assert pred_end == end
                countspans += 1
                spanstrings.append(text[start:end])

        counter += 1
        if counter == max:
            break
    assert countspans == 61
