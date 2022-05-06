# In our definition of the scorer in project.yml we use:
#    @registry.scorers("spancat_char_scorer.v1")
# This calls make_spancat_char_scorer, so we must import this to initialize our
# generated model
from ..evaluate_char import (
    evaluate_one,
    span_indexes,
    make_spancat_char_scorer,
)


import spacy
from spacy.training import Example

import statistics

import sys
sys.path.append('./scripts')

def dontdothis_test_evaluate():

    spans_key = 'txs'
    #nlp = spacy.load('en_toxic_detector')
    nlp1 = spacy.load("training/model-best")
    nlp2 = spacy.load("en_core_web_sm")

    scorer = make_spancat_char_scorer()

    is_ok = 0
    is_false = 0
    all_examples = []
    all_custom = []
    for spans, text in read_data_csv('assets/tsd_train.csv', max=10):
        # Process the text with the pipeline: pred_doc
        pred_doc = nlp1(text)

        pred_spans = pred_doc.spans[spans_key]

        # Get the pipeline from the CSV: gold_doc
        ms = make_spans(spans)
        gold_doc = nlp2(text)
        lst = []
        for start, end in ms:
            span = gold_doc.char_span(start, end, label='TOXIC')
            if span is None:
                print("broken for ", start, end, text[start:end])
            else:
                lst.append(span)
                #print('gold-doc span', span.text, 'pred-doc span', [s.text for s in pred_spans])
        gold_doc.spans[spans_key] = lst

        example = Example(pred_doc, gold_doc)

        # Evaluate the example with the standard scorer
        standard_dict = scorer([example], spans_key='txs')
        standard = (
            standard_dict['spans_' + spans_key + '_f'],
            standard_dict['spans_' + spans_key + '_p'],
            standard_dict['spans_' + spans_key + '_r'],
        )
        # Evaluate
        custom = evaluate_one(spans, pred_doc)

        if standard == custom:
            is_ok += 1
            #print('OK = ', text)
        else:
            is_false += 1
            print('****** FALSE = ', text)
            print('pred-doc spans', [s.text for s in pred_doc.spans[spans_key]])
            print('pred-doc indexes', span_indexes(pred_doc, spans_key='txs'))
            print('gold-doc spans', [s.text for s in gold_doc.spans[spans_key]])
            print('gold-doc indexes', span_indexes(gold_doc, spans_key='txs'))
            print('standard: ', standard)
            print('custom: ', custom)

        all_examples.append(example)
        all_custom.append(custom)

    total_examples = scorer(all_examples, spans_key='txs')
    #total_custom = evaluate(all_custom)

    mean_f1 = 100 * statistics.mean([f[0] for f in all_custom])
    mean_precision = 100 * statistics.mean([p[1] for p in all_custom])
    mean_recall = 100 * statistics.mean([r[2] for r in all_custom])
    print('============================== Character based metrics ============================== ')
    print(f'Precision = {mean_precision:.2f}  Recall = {mean_recall:.2f}  F1 = {mean_f1:.2f}')


    total_custom = (mean_f1, mean_precision, mean_recall)
    print(all_custom)

    print('is ok = ', is_ok)
    print('is false = ', is_false)

    print('total examples = ', total_examples)
    print('total custom = ', total_custom)

    assert total_examples == len(total_custom)

