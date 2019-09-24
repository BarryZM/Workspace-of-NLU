# coding=utf-8


def calculate_nmi(coo_value, word_value_1, word_value_2):
    return coo_value / (word_value_1 * word_value_2)


def get_prob(lines, word_list):
    total = len(lines)

    number = 0
    for line in lines:
        if set(word_list).issubset(line):  # issubset 判断list A 是否包含在 list B 中
            number += 1
    percent = float(number)/total  # 有多少个文档包含word_list / 总的文档数
    return percent


def get_coo_prob(lines, word_list1, word_list2):
    join_word_list = word_list1 + word_list2
    percent = get_prob(lines, join_word_list)
    return percent


class PMI:
    def __init__(self, lines, min_coo_prob):
        """
        :param lines: input line list after extracted word
        """
        self.lines = lines
        self.pmi = {}
        self.min_single_word_prob = float(1.0) / len(lines)
        self.min_coo_prob = min_coo_prob  # Co-occurrence probability
        self.set_word = self.get_word_list()

    def get_word_list(self):
        list_word = []
        for line in self.lines:
            list_word = list_word + list(line)

        list_word = list(set(list_word))
        return list_word

    def get_dict_frq_word(self):
        dict_frq_word = {}
        for i in range(len(self.set_word)):
            single_word_list = []
            single_word_list.append(self.set_word[i])
            probability = get_prob(self.lines, single_word_list)
            if probability > self.min_single_word_prob:
                dict_frq_word[self.set_word[i]] = probability
        return dict_frq_word

    def get_pmi(self):
        dict_pmi = {}
        dict_frq_word = self.get_dict_frq_word()  # dict {word, p(word)}
        # print(dict_frq_word)
        for word1 in dict_frq_word:
            p_word_1 = dict_frq_word[word1]
            for word2 in dict_frq_word:
                if word1 == word2:
                    continue
                p_word_2 = dict_frq_word[word2]

                coo_prob = get_prob(self.lines, [word1]+[word2])

                if coo_prob > self.min_coo_prob:
                    string = word1 + '\u001A' + word2
                    dict_pmi[string] = calculate_nmi(coo_prob, p_word_1, p_word_2)

        return dict_pmi
