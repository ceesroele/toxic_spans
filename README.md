# Detecting toxic spans with spaCy

## Introduction

Release 1.0.0 contains the code for the article ["Detecting Toxic Spans with spaCy"](https://cees-roele.medium.com/detecting-toxic-spans-with-spacy-c5533786bbf8).
Release 2.0.0 contains the code for the article ["Custom evaluation of spans in spaCy"](https://cees-roele.medium.com/custom-evaluation-of-spans-in-spacy-f1f2e7a99ad8).

An expression is toxic if it uses rude, disrespectful, or unreasonable language 
that is likely to make someone leave a discussion. Toxic language can be short 
like *"idiot"* or longer like "your 'posts' are as usual ignorant of reality".

We will use the `SpanCategorizer` from spaCy to detect toxic spans. 
For illustration we will use a well-researched dataset.

## Installation

Code uses python 3.8.

Requirement is only spaCy >= 3.1.0

If not installed yet, (create your favourite local environment for python) and do

> pip install -r requirements.txt

Also, it uses a small language model for English:

> python -m spacy download "en_core_web_sm"


## Running the code

The code is run through a spaCy "project" defined in project.yml.

Get the data:

> python -m spacy project assets

Convert data, train the model, and evaluate the best model:

> python -m spacy project run all

For separate steps:
> python -m spacy project run corpus
> python -m spacy project run train
> python -m spacy project run evaluate

To use the resulting best model:

> python run.py "Enter your toxic sentence here, my apologies for the example, you idiot"

(Don't put interpunction at the end!)

Enjoy!