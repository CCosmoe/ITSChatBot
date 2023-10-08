import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer 
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stemmy(word):
    ignore_punctuations = ['!', '?', ',', '.']                      #ignore these
    Stem_List = []
    for w in word:
        if w not in ignore_punctuations:                            #lowering all the words and stemming them
            a = stemmer.stem(w.lower())
            Stem_List.append(a)
    return Stem_List

def bag_of_words(tokenized_sentence, all_words):                    #returns all the words with list [1.0.0.0.1.....]
    tokenized_sentence = stemmy(tokenized_sentence)                
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1.0

    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bag = bag_of_words(sentence, words)
# print(bag)