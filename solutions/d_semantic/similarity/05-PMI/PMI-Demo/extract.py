# coding=utf-8
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re


def remove_emoji(sentence):
    return re.sub('\[.*?\]', '', sentence)


class Extractkeys:
    def cut_with_part_of_speech(self, sentence):
        sentence = remove_emoji(sentence)
        words =jieba.cut(sentence)
        wordlist=[]
        for word in words:
            wordlist.append(word)
        return wordlist

    def extract_word(self, word_list):
        sentence = ','.join(word_list)
        words = jieba.analyse.extract_tags(sentence, 5)
        word_list = []
        for w in words:
            word_list.append(w)
        return word_list

    def remove_stop_word(self, wordlist):
        stopWords = self.GetStopWords()
        keywords = []
        for word in wordlist:
            if word not in stopWords:
                keywords.append(word)
        return keywords


def extract(text):
    ek = Extractkeys()
    word_list = ek.cut_with_part_of_speech(text)
    extract_words = ek.extract_word(word_list)
    return extract_words


