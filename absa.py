
# coding: utf-8

# In[1]:


import re
import time
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

import en_core_web_sm
import en_coref_sm


# ## Feature sentiment extraction module based on dependency graph

# In[3]:


# dependency relation based feature sentiment extraction
def feature_sentiment(sentence):
    '''
    function: dependency relation based feature sentiment extraction
    input: dictionary and sentence
    output: updated dictionary
    '''
    sent_dict = Counter()
    sentence = spacy(sentence)
    debug = 0
#     print('token.text, token.pos_, token.dep_, token.head, token.head.dep_, token.children')
    for token in sentence:
#         print(token.text, token.pos_, token.dep_, token.head, token.head.dep_, [child for child in token.children])
        #check if the word is an opinion word, then assign sentiment
        if token.text in opinion_words:
            senti_words = []
#             sentiment = 1 if token.text in pos else -1
            if token.text in pos:
                sentiment = 1
            else:
                sentiment = -1
            senti_words = [token.text] + senti_words

            # if target is an adverb modifier (i.e. pretty, highly, etc.)
            # but happens to be an opinion word, ignore and pass
            if (token.dep_ == "advmod"):
                continue
            elif (token.dep_ == "amod"):
                # check for negation
                for child in token.head.head.children:
                    # check for negation words and flip the sign of sentiment
                    if (child.dep_ == "neg"):
                        sentiment *= -1
                        senti_words = [child.text] + senti_words
#                 sent_dict[token.head.text] += sentiment
#                 sent_dict[(token.head.text, ' '.join(senti_words))] += sentiment
                noun = token.head.text
                # check for nouns
                for child in token.head.children:
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                        noun = child.text + " " + noun
                        # Check for compound nouns
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                noun = subchild.text + " " + noun
                sent_dict[(noun, ' '.join(senti_words))] += sentiment

            # for opinion words that are adjectives, adverbs, verbs...
            else:
                for child in token.children:
                    # if there's a adj modifier (i.e. very, pretty, etc.) add more weight to sentiment
                    # This could be better updated for modifiers that either positively or negatively emphasize
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 1.5
                        senti_words = [child.text] + senti_words
                    # check for negation words and flip the sign of sentiment
                    if child.dep_ == "neg":
                        sentiment *= -1
                        senti_words = [child.text] + senti_words
                for child in token.children:
                    # if verb, check if there's a direct object
                    if (token.pos_ == "VERB") & (child.dep_ == "dobj"):
#                         sent_dict[child.text] += sentiment
                        sent_dict[(child.text, ' '.join(senti_words))] += sentiment
                        # check for conjugates (a AND b), then add both to dictionary
                        subchildren = []
                        conj = 0
                        for subchild in child.children:
                            if subchild.text == "and":
                                conj=1
                            if (conj == 1) and (subchild.text != "and"):
                                subchildren.append(subchild.text)
                                conj = 0
                        for subchild in subchildren:
#                             sent_dict[subchild] += sentiment
                            sent_dict[(subchild, ' '.join(senti_words))] += sentiment

                # check for negation
                for child in token.head.children:
                    noun = ""
                    if ((child.dep_ == "amod") or (child.dep_ == "advmod")) and (child.text in opinion_words):
                        sentiment *= 1.5
                        senti_words = [child.text] + senti_words
                    # check for negation words and flip the sign of sentiment
                    if (child.dep_ == "neg"):
                        sentiment *= -1
                        senti_words = [child.text] + senti_words

                # check for nouns
                for child in token.head.children:
                    noun = ""
                    if (child.pos_ == "NOUN") and (child.text not in sent_dict):
                        noun = child.text
                        # Check for compound nouns
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                noun = subchild.text + " " + noun
#                         sent_dict[noun] += sentiment
                        sent_dict[(noun, ' '.join(senti_words))] += sentiment
                    debug += 1
    return sent_dict


# In[4]:


def classify_and_sent(sentence):
    '''
    function: classify the sentence into a category, and assign sentiment
    input: sentence & aspect dictionary, which is going to be updated
    output: updated aspect dictionary
    note: aspect_dict is a parent dictionary with all the aspects
    '''
    # get aspect names and their sentiment in a dictionary form
    sent_dict = feature_sentiment(sentence)
    return sent_dict

def replace_pronouns(text):
    '''
    function: replaces the pronouns in a text with referring word
    input: text that needs to be processed
    output: pronoun replaced text
    '''
    doc = coref(text)
    text_updated = ''
    if doc._.coref_resolved == '':
        text_updated = text
    else:
        text_updated = doc._.coref_resolved
    return text_updated

def split_sentence(text):
    '''
    function: splits review into a list of sentences using spacy's sentence parser
    input: complete input text
    output: list of sentences
    '''
    review = spacy(text)
    bag_sentence = []
    start = 0
    for token in review:
        if token.sent_start:
            bag_sentence.append(review[start:(token.i-1)])
            start = token.i
        if token.i == len(review)-1:
            bag_sentence.append(review[start:(token.i+1)])
    return bag_sentence

def remove_special_char(sentence):
    '''
    function: removes the special characters in a text
    input: text that needs to be processed
    output: special character removed text
    '''
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', sentence)

def review_pipe(review):
    '''
    function: processing pipeline performing step by step process for aspect sentiment
    input: input review
    output: aspect sentiment list
    '''
    review = replace_pronouns(review)
    sentences = split_sentence(review)
    review_dict = Counter()
    for sentence in sentences:
        sentence = remove_special_char(str(sentence))
        sent_dict = classify_and_sent(sentence.lower())
        dict.update(review_dict,sent_dict)
    return review_dict

# feature sentiment mining for review
def review_feature_sentiment_extraction(review):
    '''
    function: extracts feature sentiments for an input review
    input: input review
    output: feature sentiment list
    '''
    feat_senti_list = []
    f_s_list = review_pipe(review).most_common()
    for f_s in f_s_list:
        f = f_s[0]
        feat = f
        # checking for exception features
        feature = f[0]
        opinion = f[1]
        if feature in exception_features:
            continue
        s = f_s[1]
        if s > 0:
            senti = 'pos'
        elif s < 0:
            senti = 'neg'
        else:
            senti = 'neu'
        feat_senti = feat, senti
        feat_senti_list.append(feat_senti)
    return feat_senti_list


# In[26]:

start_time = time.time()
# load NLP resources
spacy = en_core_web_sm.load()
coref = en_coref_sm.load()

# Load opinion lexicon
neg_file = open("resources/opinion-lexicon-English/neg_words.txt",encoding = "ISO-8859-1")
pos_file = open("resources/opinion-lexicon-English/pos_words.txt",encoding = "ISO-8859-1")
neg = [line.strip() for line in neg_file.readlines()]
pos = [line.strip() for line in pos_file.readlines()]
opinion_words = neg + pos

# Load exception feature lexicon
exception_feature_file = open("resources/exception_features.txt",encoding = "ISO-8859-1")
exception_features = [line.strip() for line in exception_feature_file.readlines()]
end_time = time.time()
execution_time = end_time - start_time
print('resource loadup time: ' + str(execution_time))


# In[9]:


# # # test code for feature sentiment
# review = "I came here with my friends on a Tuesday night. The sushi here is amazing. Our waiter was very helpful, but the music was terrible."
# # review = "This is an awesome chicken sushi."
# # review = "This is not awesome chicken soup."
# # review = "The zipper is difficult to work."


# In[28]:


# feature_sentiment_list = review_feature_sentiment_extraction(review)
# feature_sentiment_list


# ## Feature sentiment extraction from Groupon deals and transaction data

# In[29]:


# df_deal_clean = pd.read_csv('dataset/clean_deal_data.csv')
# df_review_clean = pd.read_csv('dataset/clean_review_data.csv')
# # merge review dataframe and deal dataframe
# df_transaction_clean = df_review_clean.merge(df_deal_clean, left_on=['dealid', 'deal_url'], right_on=['dealid', 'deal_url'])
# df = df_transaction_clean[0:5]

df_transaction_clean = pd.read_csv('input/data_annotated.csv')
df_sample = df_transaction_clean[['dealid','reviewid','review_text','feature_sentiment_annotated']]

df_sample['review_text'] = df_sample.review_text.astype(str)
start_time = time.time()
df_sample['feature_sentiment_extracted'] = df_sample['review_text'].apply(review_feature_sentiment_extraction)
end_time = time.time()
execution_time = end_time - start_time
print('execution time: ' + str(execution_time))


df_sample.to_csv('output/transaction_feature_sentiment_data.csv', index=False)
