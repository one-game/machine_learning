#!coding=utf-8
############################
#
#
#
#
#
#################################
from enum import Enum

class TriTreeState(Enum):
    FOUND = 1
    NOT_FOUND_WITH_NEXT = 2
    NOT_FOUND_LAST_ERROR = 3

class TriTree:

    def __init__(self):
        self.__root = {}
        self.__end = "*"
        self.__num_words = 0

    def add_word(self, word, word_info = None):
        temp_tree = self.__root
        for char in word:
            if char in temp_tree:
                temp_tree = temp_tree[char]
            else:
                temp_tree = temp_tree.setdefault(char,{})
        
        temp_tree[self.__end] = word_info
        self.__num_words += 1

    def get_word_cnt(self):
        return self.__num_words

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    
    def find_word(self, word, detail = {}):
        temp_tree = self.__root
        for char in word:
            if char in temp_tree:
                temp_tree = temp_tree[char]
            else:
                return TriTreeState.NOT_FOUND_LAST_ERROR
        else:
            if self.__end in temp_tree:
                detail["detail"] = temp_tree[self.__end]
                return TriTreeState.FOUND
            else:
                return TriTreeState.NOT_FOUND_WITH_NEXT

    def match_all_words(self, sentence):
        sent_len = len(sentence)
        word_condidates = []
        for first_ind in range(sent_len):
            for end_ind in range(first_ind + 1, sent_len + 1):
                candidate = sentence[first_ind:end_ind]
                
                ret_state = self.find_word(candidate)
                #print first_ind, ":", end_ind, "=>look up ", candidate, " ret_state is ", ret_state
                if ret_state  == TriTreeState.FOUND:
                    word_condidates.append((candidate, first_ind, end_ind))
                elif ret_state == TriTreeState.NOT_FOUND_LAST_ERROR:
                    break
        return word_condidates

    


if __name__ == "__main__":
    triTree = TriTree()
    #triTree.add_words([u"我",u"我爱",u"z中国",u"中央",u"中央人民政府"])
    #print triTree.find_word(u"中央")
    import json
    import time
    start_time = time.time()
    filename = "/Users/billzhang/Documents/work/code/third/chinese-xinhua/data/ci.json"
    with open(filename, "r") as fd:
        for j_data in json.loads(fd.read()):
            triTree.add_word( j_data["ci"], j_data)
    end_time = time.time()
    print triTree.get_word_cnt()
    print "loading dict cost is %fms"%(end_time - start_time)
    detail = {}
    print triTree.find_word(u"政府", detail)
    print detail
    #print ",".join([word for word, _, _ in triTree.match_all_words(u"我爱中国中央人民政府")])
