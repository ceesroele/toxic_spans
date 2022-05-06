from typing import Optional, Iterable, Dict, Set, Any, Callable, Tuple
from typing import TYPE_CHECKING

from spacy.training import Example
from spacy.tokens import Token, Doc, Span
from spacy.util import registry

import statistics

if TYPE_CHECKING:
    # This lets us add type hints for mypy etc. without causing circular imports
    from spacy.language import Language  # noqa: F401

from util import span2char

DEFAULT_PIPELINE = ("senter", "tagger", "morphologizer", "parser", "ner", "textcat")
MISSING_VALUES = frozenset([None, 0, ""])


def spancat_char_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    kwargs = dict(kwargs)
    attr_prefix = "spans_"
    key = kwargs["spans_key"]
    kwargs.setdefault("attr", f"{attr_prefix}{key}")
    kwargs.setdefault("allow_overlap", True)
    kwargs.setdefault(
        "getter", lambda doc, key: doc.spans.get(key[len(attr_prefix) :], [])
    )
    kwargs.setdefault("has_annotation", lambda doc: key in doc.spans)
    return score_spans_char(examples, **kwargs)


@registry.scorers("spancat_char_scorer.v1")
def make_spancat_char_scorer():
    return spancat_char_score


class PRFCharScore:
    """A character-based precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.f1_list = []
        self.p_list = []
        self.r_list = []

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFCharScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

        tp = len(cand.intersection(gold))
        fp = len(cand - gold)
        fn = len(gold - cand)

        if len(cand) == 0:
            cur_p = 1.0 if len(gold) == 0 else 0.0
        else:
            cur_p = tp / (tp + fp + 1e-100)

        if len(gold) == 0:
            cur_r = 1.0 if len(cand) == 0 else 0.0
        else:
            if tp + fn == 0:
                cur_r = 0.0
            else:
                cur_r = tp / (tp + fn)

            #cur_r = tp / (tp + fn + 1e-100)

        #if tp == fp == fn == 0:
        #    cur_f1 = 1.0
        if len(gold) == 0:
            cur_f1 = 1.0 if len(cand) == 0 else 0.0
        else:
            #cur_f1 = 2 * ((cur_p * cur_r) / (cur_p + cur_r + 1e-100))
            nom = 2 * len(cand.intersection(gold))
            denom = len(cand) + len(gold)
            cur_f1 = nom / denom

        self.f1_list.append(cur_f1)
        self.p_list.append(cur_p)
        self.r_list.append(cur_r)


    @property
    def precision_orig(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall_orig(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def fscore_orig(self) -> float:
        # change: empty candidate and empty gold is a score of 1.0
        if self.tp == self.fp == self.fn:
            return 1.0

        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def precision(self) -> float:
        return statistics.mean(self.p_list)

    @property
    def recall(self) -> float:
        return statistics.mean(self.r_list)

    @property
    def fscore(self) -> float:
        return statistics.mean(self.f1_list)

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f": self.fscore}


def score_spans_char(
    examples: Iterable[Example],
    attr: str,
    *,
    getter: Callable[[Doc, str], Iterable[Span]] = getattr,
    has_annotation: Optional[Callable[[Doc], bool]] = None,
    labeled: bool = True,
    allow_overlap: bool = False,
    **cfg,
) -> Dict[str, Any]:
    """Returns PRF char-based scores for labeled spans.
    examples (Iterable[Example]): Examples to score
    attr (str): The attribute to score.
    getter (Callable[[Doc, str], Iterable[Span]]): Defaults to getattr. If
        provided, getter(doc, attr) should return the spans for the
        individual doc.
    has_annotation (Optional[Callable[[Doc], bool]]) should return whether a `Doc`
        has annotation for this `attr`. Docs without annotation are skipped for
        scoring purposes.
    labeled (bool): Whether or not to include label information in
        the evaluation. If set to 'False', two spans will be considered
        equal if their start and end match, irrespective of their label.
    allow_overlap (bool): Whether or not to allow overlapping spans.
        If set to 'False', the alignment will automatically resolve conflicts.
    RETURNS (Dict[str, Any]): A dictionary containing the PRF scores under
        the keys attr_p/r/f and the per-type PRF scores under attr_per_type.
    DOCS: https://spacy.io/api/scorer#score_spans
    """
    score = PRFCharScore()
    score_per_type = dict()
    #print("HANDLING ",len(examples), "examples")
    for example in examples:
        pred_doc = example.predicted
        gold_doc = example.reference
        #print(gold_doc.text)
        # Option to handle docs without annotation for this attribute
        if has_annotation is not None and not has_annotation(gold_doc):
            continue
        # Find all labels in gold
        labels = set([k.label_ for k in getter(gold_doc, attr)])
        # If labeled, find all labels in pred
        if has_annotation is None or (
            has_annotation is not None and has_annotation(pred_doc)
        ):
            labels |= set([k.label_ for k in getter(pred_doc, attr)])
        # Set up all labels for per type scoring and prepare gold per type
        gold_per_type: Dict[str, Set] = {label: set() for label in labels}
        for label in labels:
            if label not in score_per_type:
                score_per_type[label] = PRFCharScore()
        # Find all predicate labels, for all and per type
        gold_spans = set()
        pred_spans = set()
        for span in getter(gold_doc, attr):
            gold_span: Tuple
            if labeled:
                #gold_span = (span.label_, span.start, span.end - 1)
                gold_span = (span.label_,) + span2char(gold_doc, span)
            else:
                #gold_span = (span.start, span.end - 1)
                gold_span = span2char(gold_doc, span)
            gold_spans.add(gold_span)
            gold_per_type[span.label_].add(gold_span)
        pred_per_type: Dict[str, Set] = {label: set() for label in labels}
        if has_annotation is None or (
            has_annotation is not None and has_annotation(pred_doc)
        ):
            for span in example.get_aligned_spans_x2y(
                getter(pred_doc, attr), allow_overlap
            ):
                pred_span: Tuple
                if labeled:
                    #pred_span = (span.label_, span.start, span.end - 1)
                    pred_span = (span.label_,) + span2char(pred_doc, span)
                    #print('prf_char_scorer:', pred_span, span.text)
                else:
                    #pred_span = (span.start, span.end - 1)
                    print('Not labeled span')
                    pred_span = span2char(pred_doc, span)
                pred_spans.add(pred_span)
                pred_per_type[span.label_].add(pred_span)
        # Scores per label
        if labeled:
            for k, v in score_per_type.items():
                if k in pred_per_type:
                    v.score_set(pred_per_type[k], gold_per_type[k])
        # Score for all labels
        score.score_set(pred_spans, gold_spans)
    # Assemble final result
    final_scores: Dict[str, Any] = {
        f"{attr}_p": None,
        f"{attr}_r": None,
        f"{attr}_f": None,
    }
    if labeled:
        final_scores[f"{attr}_per_type"] = None
    #print(f'SCORES: tp={score.tp}, fp={score.fp}, fn={score.fn}')
    #print('*'*70)
    if len(score) > 0:
        final_scores[f"{attr}_p"] = score.precision
        final_scores[f"{attr}_r"] = score.recall
        final_scores[f"{attr}_f"] = score.fscore
        if labeled:
            final_scores[f"{attr}_per_type"] = {
                k: v.to_dict() for k, v in score_per_type.items()
            }
    return final_scores
