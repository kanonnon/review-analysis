import re
import os
import MeCab


def get_japanese_stop_words():
    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, "data/Japanese.txt")
    with open(file_path,"r") as f:
        stop_words = f.read().split("\n")
    return stop_words


def tokenize_and_filter_sentence(input_sentence):
    lower_case_sentence = input_sentence.lower()
    clean_sentence = re.sub(r'[【】]|[（）()]|[［］\[\]]|[@＠]\w+|\d+\.*\d*', ' ', lower_case_sentence)

    mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    lines = mecab.parse(clean_sentence).split("\n")[:-2]

    words = [line.split("\t")[1].split(",")[6] for line in lines]
    parts_of_speech = [line.split('\t')[1].split(",")[0] for line in lines]
    nouns = [word for word, pos in zip(words, parts_of_speech) if pos == "名詞"]
    
    stop_words = get_japanese_stop_words()
    filtered_words = [noun for noun in nouns if noun not in stop_words]
    
    return filtered_words
