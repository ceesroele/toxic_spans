import csv
import ast
import pandas as pd
from spacy.tokens import Doc, Span


def read_data_csv(fname: str, max=None):
    """
    Read data from CSV file with rows (list(indexes), text)
    :param fname: (relative) path of file with CSV data
    """
    counter = 0
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader)  # skip the headers
        for row in reader:
            #print(row)
            lst = ast.literal_eval(row[0])
            text = row[1]
            counter += 1
            if max is not None and counter > max:
                break
            yield lst, text


def read_data_df(fname: str, max:int=None):
    df = pd.read_csv(fname)
    df.spans = df.spans.apply(ast.literal_eval)
    counter = 0
    for span, text in zip(df.spans.to_list(), df.text.to_list()):
        counter += 1
        if max is not None and counter > max:
            break
        yield span, text


def read_data(fname: str, max=None):
    return read_data_csv(fname, max=max)


if __name__ == '__main__':
    for lst, text in read_data('assets/tsd_train.csv'):
        spans = make_spans(lst)
        for s, e in spans:
            print(text[s:e])