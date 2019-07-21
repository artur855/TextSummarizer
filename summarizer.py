from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk import FreqDist
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import numpy as np
import networkx as nx


def debug_text():
    with open('text.txt') as txt:
        return '\n'.join(txt.readlines())


def sentence_simmilarity(s1, s2):
    return 1 - cosine_distance(s1, s2)


def summarize_text(text, n=None):
    stopwords = nltk.corpus.stopwords.words('portuguese')

    vector = TfidfVectorizer()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    sentences = sent_tokenize(text)

    fix_sentences = []
    for sentence in sentences:
        new_sentence = []
        for word in tokenizer.tokenize(sentence):
            word = word.lower()
            if word in stopwords or word in string.punctuation:
                continue
            new_sentence.append(word)
        fix_sentences.append(' '.join(new_sentence))

    vector.fit(fix_sentences)

    word_embeddings = [vector.transform([sentence]).toarray()[0]
                       for sentence in fix_sentences]

    shape = (len(word_embeddings), len(word_embeddings))
    matrix = np.zeros(shape)

    for i, s1 in enumerate(word_embeddings):
        for j, s2 in enumerate(word_embeddings):
            if i != j:
                matrix[i][j] = sentence_simmilarity(s1, s2)

    graph = nx.from_numpy_array(matrix)
    rank = nx.pagerank(graph)
    rank = [(rank[i], s) for i, s in enumerate(sentences)]
    rank.sort(key=lambda x: x[0], reverse=True)
    rank = [tp[1] for tp in rank]
    if not n:
        n = min(len(rank), 2)
    return '\n'.join(rank[:n])


if __name__ == "__main__":
    text = debug_text()
    summary = summarize_text(text)
    print('===================== Texto sumarizado  =====================')
    print(summary)
    rate = len(summary) / len(text)
    rate = 1 - rate
    print('Texto resumido em {:.0f}%'.format(rate*100))
