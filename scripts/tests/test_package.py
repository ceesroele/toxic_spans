""""
Test deployed package
"""
import spacy

def test_loadable():
    """Test whether the created and deployed package is loadable"""
    nlp = spacy.load('en_toxic_detector')
    doc = nlp('hello, world!')
    assert doc is not None