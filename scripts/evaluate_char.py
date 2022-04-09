"""
Evaluate in terms of F1-score according to character-based similarity between gold spans
and predicted spans.

Note that the standard PRFScore (Precision, Recall, F) of spaCy is based on the index of
tokens in a span, rather than the characters.

The reason for a character-based score is that this is the method used by the SemEval2021
"Toxic Spans Prediction" task. By using this method, we can compare our outcome with those
of the published articles discussing results of the SemEval2021 task.
"""
import spacy
import ast
import statistics
from util import read_data


def token_to_index(doc):
    """List of (start,end) tuples for all tokens in a doc (excluding trailing whitespace)"""
    lst = []
    cur = 0
    for t in doc:
        tlength = len(t.text)
        tlength_with_ws = len(t.text_with_ws)
        lst.append((cur, cur + tlength))
        cur += tlength_with_ws
    return lst


def span_indexes(doc, spans_key='sc'):
    """List of character indexes of every matching character in every span"""
    tok2index = token_to_index(doc)
    lst = []
    for span in doc.spans[spans_key]:
        start = tok2index[span.start][0]
        end = tok2index[span.end - 1][1]
        for i in range(start, end):
            lst.append(i)
    matching_words = []
    text = doc.text_with_ws
    for i in lst:
        matching_words.append(text[i])
    return lst


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1.0 if len(predictions) == 0 else 0.0
    nom = 2*len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions))+len(set(gold))
    return nom/denom


def precision(predictions, gold):
    """
    Precision operating on two lists of offsets (e.g., character).
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(predictions) == 0:
        return 1.0 if len(gold) == 0 else 0.0
    else:
        tp = len(set(predictions).intersection(set(gold)))
        fp = len(set(predictions) - set(gold))
        return tp / (tp + fp)


def recall(predictions, gold):
    """
    Recall operating on two lists of offsets (e.g., character).
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1.0 if len(predictions) == 0 else 0.0
    else:
        tp = len(set(predictions).intersection(set(gold)))
        fn = len(set(gold) - set(predictions))
        if tp + fn == 0:
            return 0.0
        else:
            return tp / (tp + fn)


def evaluate():
    """Character index based evaluation"""
    nlp = spacy.load("training/model-best")
    f1_list = []
    precision_list = []
    recall_list = []
    for lst, text in read_data('assets/tsd_test.csv'):
        doc = nlp(text)
        outcome = span_indexes(doc, spans_key='txs')
        f1_res = f1(outcome, lst)
        f1_list.append(f1_res)
        precision_res = precision(outcome, lst)
        precision_list.append(precision_res)
        recall_res = recall(outcome, lst)
        recall_list.append(recall_res)
    mean_precision = 100 * statistics.mean(precision_list)
    mean_recall = 100 * statistics.mean(recall_list)
    mean_f1 = 100 * statistics.mean(f1_list)
    print('============================== Character based metrics ============================== ')
    print(f'Precision = {mean_precision:.2f}  Recall = {mean_recall:.2f}  F1 = {mean_f1:.2f}')


if __name__ == '__main__':
    evaluate()
