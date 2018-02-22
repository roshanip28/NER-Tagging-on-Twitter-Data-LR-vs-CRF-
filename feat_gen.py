#!/bin/python
import os
import string
from collections import defaultdict
import nltk


d = defaultdict(list)
stops_words = []


def ws(chars):
    if chars.islower():
        return 'x'
    elif chars.isupper():
        return 'X'
    elif chars.isdigit():
        return 'd'
    else:
        return chars

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    f = open("data/lexicon/english.stop")
    for name in f:
        name = name.strip()
        stops_words.append(name)
    f.close()


    for filename in os.listdir("data/lexicon/"):
        f = open("data/lexicon/" + filename)
        if not (filename == "internet.website" or filename == "automotive.model" or filename == "dictionaries.conf" or filename == "venues"):
            for x in f.readlines():
                y = x.strip().lower()
                y = ''.join(ch for ch in y if ch not in string.punctuation)

                listy = y.split()
                for xy in listy:
                    if xy != "":
                        d[xy].append(filename)

    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ct = 0

    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent) - 1:
        ftrs.append("SENT_END")
    # the word itself
    word = unicode(sent[i])
    #vov=['a','e','i','o','u','A','E','I','O','U']
    vov="aeiouAEIOU"
    for a in word:
        if a in vov:
            ct += 1

    ftrs.append("#v" + str(ct / len(sent)))

    y = word.strip().lower()
    simple_string = ''.join(ch for ch in y if ch not in string.punctuation)

    if simple_string in d:
        if simple_string != "":
            for j in d[simple_string]:
                if j not in ftrs:
                    ftrs.append(j)


    stri = ""
    for i in range(len(word) - 2):
        if i < 2:
            stri += ws(word[i])
        elif i >= 2:
            if stri[-1] != ws(word[i]):
                stri += ws(word[i])
    if len(word) > 2:
        stri += ws(word[-2])
    if len(word) > 3:
        stri += ws(word[-1])

    ftrs.append(stri + ": WS")

    y = nltk.pos_tag(sent)
    x = y[i]
    ftrs.append(x[1])
    #








    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
