import os
import sys

import numpy as np
from tqdm import tqdm
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import CountVectorizer

from util import tokenize_and_filter_sentence


def get_file_path(file, directory_name):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    if file == "input":
        file_path = os.path.join(current_directory, "data/{}/review.txt".format(directory_name))
        print(file_path)
    elif file == "output":
        file_path = os.path.join(current_directory, "data/{}/top_hundred_words.txt".format(directory_name))
    return file_path


def get_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences


def detect_collocations(noun_words):
    phrases = Phrases(noun_words, min_count=1, threshold=1)
    phraser = Phraser(phrases)
    noun_collocation = [phraser[sentence] for sentence in noun_words]
    return noun_collocation


def calculate_tf(noun_collocation):
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r"(?u)\b\w+\b")
    tf_matrix = vectorizer.fit_transform([' '.join(sentence) for sentence in noun_collocation])
    total_term_count = np.sum(tf_matrix)
    terms = vectorizer.get_feature_names()
    tfs = np.round(tf_matrix.sum(axis=0).getA1() / total_term_count, 4)
    return tfs, terms


def get_top_hundred_terms(tfs, terms):
    high_tf_idxs = np.argsort(tfs)[::-1]
    top_hundred_idxs = high_tf_idxs[:100]
    return [(terms[idx], tfs[idx]) for idx in top_hundred_idxs]


def write_to_text(filename, top_hundred_terms):
    with open(filename, 'w', encoding='utf-8') as f:
        for term, tf in top_hundred_terms:
            f.write(f"{term}: {tf}\n")


def main(directory_name):
    sentences = get_sentences(get_file_path("input", directory_name))
    filtered_words = [tokenize_and_filter_sentence(sentence) for sentence in tqdm(sentences)]
    noun_collocation = detect_collocations(filtered_words)
    tfs, terms = calculate_tf(noun_collocation)
    top_hundred_terms = get_top_hundred_terms(tfs, terms)
    write_to_text(get_file_path("output", directory_name), top_hundred_terms)


if __name__ == '__main__':
    directory_name = sys.argv[1]
    main(directory_name)
