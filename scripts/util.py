import csv
import ast


def read_data(fname: str):
    """
    Read data from CSV file with rows (list(indexes), text)
    :param fname: (relative) path of file with CSV data
    """
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader)  # skip the headers
        for row in reader:
            lst = ast.literal_eval(row[0])
            text = row[1]
            yield lst, text


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


if __name__ == '__main__':
    for lst, text in read_data('assets/tsd_train.csv'):
        spans = make_spans(lst)
        for s, e in spans:
            print(text[s:e])