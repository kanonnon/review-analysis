import os
import sys

import pandas as pd
from tqdm import tqdm
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from util import tokenize_and_filter_sentence


X = 10
Y = 0.5
NUM_TOPICS = 6
NUM_TOP_WORDS = 15


def get_file_path(file, directory_name):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    if file == "input":
        file_path = os.path.join(current_directory, "data/{}/review.txt".format(directory_name))
    elif file == "output":
        file_path = os.path.join(current_directory, "data/{}/topic_words.csv".format(directory_name))
    return file_path


def get_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences


def create_dictionary(filtered_words):
    dictionary = Dictionary(filtered_words)
    dictionary.filter_extremes(no_below=X, no_above=Y)
    return dictionary


def lda_model(filtered_words, dictionary):
    corpus = [dictionary.doc2bow(text) for text in filtered_words]
    lda = LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS, alpha=0.01)
    return lda


def create_df(lda, dictionary):
    df = pd.DataFrame()
    for t in range(NUM_TOPICS):
        word=[]
        for i, prob in lda.get_topic_terms(t, topn=NUM_TOP_WORDS):
            word.append(dictionary.id2token[int(i)])
        _ = pd.DataFrame([word],index=[f'topic{t+1}'])
        df = df.append(_)
    return df


def write_to_csv(df, directory_name):
    file_path = get_file_path("output", directory_name)
    df.to_csv(file_path, encoding='utf-8')
    return


def main(directory_name):
    sentences = get_sentences(get_file_path("input", directory_name))
    filtered_words = [tokenize_and_filter_sentence(sentence) for sentence in tqdm(sentences)]
    dictionary = create_dictionary(filtered_words)
    lda = lda_model(filtered_words, dictionary)
    df = create_df(lda, dictionary)
    write_to_csv(df, directory_name)


if __name__ == '__main__':
    directory_name = sys.argv[1]
    main(directory_name)
