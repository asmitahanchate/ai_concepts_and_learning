from ragkit.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords_keeps_identifiers():
    assert tokenize("How do I use the convert_underscores flag?") == ["use", "convert_underscores", "flag"]


def test_rare_word_wins():
    texts = ["the api returns json", "the api uses a threadpool for def functions", "json json json"]
    index = BM25(texts)
    assert index.search("threadpool", k=1)[0][0] == 1  # only doc 1 has the rare word
