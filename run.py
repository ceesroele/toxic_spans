"""
Run the resulting model using a little script
taking a sentence as command line argument
"""

import spacy
import sys
import argparse


def run(s: str, spans_key='sc', model_reference='training/model-best'):
    """"""
    # Load the best model resulting from the training
    # nlp = spacy.load('training/model-best')
    # Or load the pip installed version inlcuding an additional script
    # nlp = spacy.load('en_toxic_detector')
    nlp = spacy.load(model_reference)
    doc = nlp(s)
    spans = doc.spans[spans_key]
    if not spans:
        print('No toxic spans found in: ', s)
    else:
        print('Toxic span(s):')
        spanlst = []
        for span in spans:
            t = (span.start, span.end, span.label_)
            spanlst.append(t)
            print(f'- {span.text} {t}')
        outcome = []
        in_span = False
        for token in doc:
            index = token.i
            for start, end, _ in spanlst:
                if index == start:
                    outcome.append('\033[35m')
                    in_span = True
                if index == end:
                    top = outcome.pop()
                    outcome.append(top[:-1])
                    outcome.append('\033[00m ')
                    in_span = False
            outcome.append(token.text_with_ws)
        if in_span:
            outcome.append('\033[00m ')
        print(''.join(outcome))


if __name__ == '__main__':
    spans_key = 'txs'

    parser = argparse.ArgumentParser(description='Detect toxic language in text.')
    parser.add_argument('--model', type=str, dest='model_reference',
                        default='training/model-best',
                        help='Reference to the spaCy model to be loaded. \
                        Either directory or name of installed model')
    parser.add_argument('input', help='Enter sentence for which toxic language is to be detected')
    args = parser.parse_args()
    run(args.input, spans_key=spans_key, model_reference=args.model_reference)
