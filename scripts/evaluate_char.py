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
import statistics

import sys
sys.path.append('./scripts')

import json

import typer
from typing import Optional, Iterable, Dict, Set, Any, Callable, Tuple

from util import read_data
from spacy.tokens import DocBin, Doc, Span
from spacy.util import registry
from spacy.training import Example


def make_spans(lst):
    """Split list of numbers into a list of (start,end) tuples,
    e.g. [1,2,3,4,10,11,12] of indexes becomes [(1,5), (10,13)] (exclusive end index)"""
    start = None
    spans = []
    last_d = None
    for d in lst:
        if last_d is None:
            start = d
        elif d > last_d + 1:
            spans.append( (start, last_d + 1) )
            start = d
        last_d = d
    if start is not None:
        spans.append( (start, last_d + 1) )
    return spans


def span2char(doc: Doc, span: Span):
    """Character index for span. End is last index plus 1"""
    prefix = ''
    start = span.start
    end = span.end

    span_text = ''
    for i, token in enumerate(doc):
        if i < start:
            prefix += token.text_with_ws
        elif start <= i < end - 1:
            span_text += token.text_with_ws
        elif i == end - 1:
            span_text += token.text

    res_start = len(prefix)
    res_end = res_start + len(span_text)
    return res_start, res_end


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



# In our definition of the scorer in project.yml we wrote:
#    scorer = {"@scorers":"spancat_char_scorer.v1"}
# This refers to a method we register in our code:
#    @registry.scorers("spancat_char_scorer.v1")
# This defines a method without arguments that returns a function which does have arguments
# to actually do the scoring. Note that this method must be in scope to load our generated model.
@registry.scorers("spancat_char_scorer.v1")
def make_spancat_char_scorer():
    return spancat_char_score


# We merge `Scorer.spancat_score` and `Scorer.score_spans` from the spaCy source code
# and ignore a number of features.
# https://github.com/explosion/spaCy/blob/master/spacy/scorer.py
def spancat_char_score(
    examples: Iterable[Example],
    **cfg,
) -> Dict[str, Any]:
    """Returns PRF char-based scores for labeled spans.
    examples (Iterable[Example]): Examples to score
    **cfg: requires keyword: spans_key
    RETURNS (Dict[str, Any]): A dictionary containing the PRF scores under
        the keys attr_p/r/f
    DOCS: https://spacy.io/api/scorer#score_spans
    """
    f1_list = []
    precision_list = []
    recall_list = []
    spans_key = cfg['spans_key']
    for example in examples:
        pred_doc = example.predicted
        gold_doc = example.reference

        gold_spans = span_indexes(gold_doc, spans_key=spans_key)
        pred_spans = span_indexes(pred_doc, spans_key=spans_key)

        f1_list.append(f1(pred_spans, gold_spans))
        precision_list.append(precision(pred_spans, gold_spans))
        recall_list.append(recall(pred_spans, gold_spans))

    mean_precision = statistics.mean(precision_list)
    mean_recall = statistics.mean(recall_list)
    mean_f1 = statistics.mean(f1_list)

    final_scores: Dict[str, Any] = {
        f"spans_{spans_key}_p": mean_precision,
        f"spans_{spans_key}_r": mean_recall,
        f"spans_{spans_key}_f": mean_f1,
    }
    return final_scores


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


def evaluate_one(gold_spans, pred_doc):
    pred_spans = span_indexes(pred_doc, spans_key='txs')

    f1_res = f1(pred_spans, gold_spans)
    precision_res = precision(pred_spans, gold_spans)
    recall_res = recall(pred_spans, gold_spans)

    return f1_res, precision_res, recall_res


def generate_evaluation():
    """Generate tuples of (spans, text, pred_doc), where spans and text come from CSV """
    nlp = spacy.load("training/model-best")
    # note that tsd_test.csv is the evaluation set
    for spans, text in read_data('assets/tsd_test.csv'):
        doc = nlp(text)
        yield spans, text, doc


def custom_evaluation(model_path: str, data_path: str, spans_key: str = 'sc', code: str = None,
                      output: str = None, gpu_id: int = -1, batch_size: int = 128):
    # The arguments are now available as positional CLI arguments
    def make_examples(data_path):
        nlp = spacy.load(model_path)
        docbin = DocBin().from_disk(data_path)
        for gold_doc in docbin.get_docs(nlp.vocab):
            pred_doc = nlp(gold_doc.text)
            yield Example(pred_doc, gold_doc)

    d = make_spancat_char_scorer()(make_examples(data_path), spans_key=spans_key)

    with open(output, 'w') as f:
        json.dump(d, f)

    mean_precision = d[f'spans_{spans_key}_p']
    mean_recall = d[f'spans_{spans_key}_r']
    mean_f1 = d[f'spans_{spans_key}_f']

    print('============================== Character based metrics ============================== ')
    print(f'Precision = {mean_precision:.3f}  Recall = {mean_recall:.3f}  F1 = {mean_f1:.3f}')


if __name__ == '__main__':
    typer.run(custom_evaluation)
