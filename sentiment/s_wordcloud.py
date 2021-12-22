import numpy as np
import pandas as pd 

import os


import wordcloud

import nltk

import re
import random
import math
# from tqdm.notebook import tqdm
from collections import Counter


from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt 

import wordninja
from spellchecker import SpellChecker
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))  
stop_words.add("amp")


import pandas as pd

data = pd.read_csv("../data/vaccine/oxford/oxford_result.csv")
# data contains one non-string entry for 'text'
str_mask = [isinstance(x, str) for x in data.clean_tweet]
data = data[str_mask]

# standard tweet preprocessing 

data.clean_tweet =data.clean_tweet.str.lower()
#Remove twitter handlers
data.clean_tweet = data.clean_tweet.apply(lambda x:re.sub('@[^\s]+','',x))
#remove hashtags
data.clean_tweet = data.clean_tweet.apply(lambda x:re.sub(r'\B#\S+','',x))
# Remove URLS
data.clean_tweet = data.clean_tweet.apply(lambda x:re.sub(r"http\S+", "", x))
# Remove all the special characters
data.clean_tweet = data.clean_tweet.apply(lambda x:' '.join(re.findall(r'\w+', x)))
#remove all single characters
data.clean_tweet = data.clean_tweet.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
# Substituting multiple spaces with single space
data.clean_tweet = data.clean_tweet.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
# Convert string to a list of words
data['words'] = data.clean_tweet.apply(lambda x:re.findall(r'\w+', x ))

# Helper functions 
def get_sign(x, p, n):
    if x > p:
        return 1
    if x < n:
        return -1 
    return 0

def flatten_list(l):
    return [x for y in l for x in y]


import nltk
nltk.download('vader_lexicon')
sia = SIA()

sentiments = [sia.polarity_scores(x)['compound'] for x in tqdm(data['clean_tweet'])]
classes = [get_sign(s, 0.35, -0.05) for s in sentiments]
data['classes'] = classes


def is_acceptable(word: str):
    return word not in stop_words and len(word) > 2


# Create one document each for all words in the negative, neutral and  positive classes respectively
neg_doc = flatten_list(data[data['classes'] == -1]['words'])
neg_doc = [x for x in neg_doc if is_acceptable(x)]

pos_doc = flatten_list(data[data['classes'] == +1]['words'])
pos_doc = [x for x in pos_doc if is_acceptable(x)]

neu_doc = flatten_list(data[data['classes'] == 0]['words'])
neu_doc = [x for x in neu_doc if is_acceptable(x)]



# color coding our wordclouds 
def red_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return f"hsl(0, 100%, {random.randint(25, 75)}%)" 

def green_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return f"hsl({random.randint(90, 150)}, 100%, 30%)" 

def yellow_color_func(word, font_size, position, orientation, random_state=None,**kwargs):
    return f"hsl(42, 100%, {random.randint(25, 50)}%)" 


# reusable function to generate word clouds 
def generate_word_clouds(neg_doc, neu_doc, pos_doc,name):
    # Display the generated image:
    fig, axes = plt.subplots(1,3, figsize=(20,10))
    
    
    wordcloud_neg = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(neg_doc))
    axes[0].imshow(wordcloud_neg.recolor(color_func=red_color_func, random_state=3), interpolation='bilinear')
    axes[0].set_title("Negative Tweets")
    axes[0].axis("off")

    wordcloud_neu = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(neu_doc))
    axes[1].imshow(wordcloud_neu.recolor(color_func=yellow_color_func, random_state=3), interpolation='bilinear')
    axes[1].set_title("Neutral Words")
    axes[1].axis("off")

    wordcloud_pos = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(pos_doc))
    axes[2].imshow(wordcloud_pos.recolor(color_func=green_color_func, random_state=3), interpolation='bilinear')
    axes[2].set_title("Positive Words")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show();
    #plt.savefig(name+'wordcloud.png')


# Naive word clouds 
generate_word_clouds(neg_doc, neu_doc, pos_doc, "oxford-naive")


def get_top_percent_words(doc, percent):
    # returns a list of "top-n" most frequent words in a list 
    top_n = int(percent * len(set(doc)))
    counter = Counter(doc).most_common(top_n)
    top_n_words = [x[0] for x in counter]
    
    return top_n_words
    
def clean_document(doc):
    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize words (needed for calculating frequencies correctly )
    doc = [lemmatizer.lemmatize(x) for x in doc]
    
    # get the top 10% of all words. This may include "misspelled" words 
    top_n_words = get_top_percent_words(doc, 0.1)

    # get a list of misspelled words 
    misspelled = spell.unknown(doc)
    
    # accept the correctly spelled words and top_n words 
    clean_words = [x for x in doc if x not in misspelled or x in top_n_words]
    
    # try to split the misspelled words to generate good words (ex. "lifeisstrange" -> ["life", "is", "strange"])
    words_to_split = [x for x in doc if x in misspelled and x not in top_n_words]
    split_words = flatten_list([wordninja.split(x) for x in words_to_split])
    
    # some splits may be nonsensical, so reject them ("llouis" -> ['ll', 'ou', "is"])
    clean_words.extend(spell.known(split_words))
    
    return clean_words


def get_log_likelihood(doc1, doc2):    

    doc1_counts = Counter(doc1)
    doc1_freq = {
        x: doc1_counts[x]/len(doc1)
        for x in doc1_counts
    }
    
    doc2_counts = Counter(doc2)
    doc2_freq = {
        x: doc2_counts[x]/len(doc2)
        for x in doc2_counts
    }
    
    doc_ratios = {
        # 1 is added to prevent division by 0
        x: math.log((doc1_freq[x] +1 )/(doc2_freq[x]+1))
        for x in doc1_freq if x in doc2_freq
    }
    
    top_ratios = Counter(doc_ratios).most_common()
    top_percent = int(0.1 * len(top_ratios))
    return top_ratios[:top_percent]



import nltk
nltk.download('wordnet')
# clean all the documents
neg_doc_clean = clean_document(neg_doc)
neu_doc_clean = clean_document(neu_doc)
pos_doc_clean = clean_document(pos_doc)

# combine classes B and C to compare against A (ex. "positive" vs "non-positive")
top_neg_words = get_log_likelihood(neg_doc_clean, flatten_list([pos_doc_clean, neu_doc_clean]))
top_neu_words = get_log_likelihood(neu_doc_clean, flatten_list([pos_doc_clean, neg_doc_clean]))
top_pos_words = get_log_likelihood(pos_doc_clean, flatten_list([neu_doc_clean, neg_doc_clean]))


top_neg_words[:5]
top_neu_words[:5]
top_pos_words[:5]

# function to generate a document based on likelihood values for words 
def get_scaled_list(log_list):
    counts = [int(x[1]*100000) for x in log_list]
    words = [x[0] for x in log_list]
    cloud = []
    for i, word in enumerate(words):
        cloud.extend([word]*counts[i])
    # shuffle to make it more "real"
    random.shuffle(cloud)
    return cloud


# Generate syntetic a corpus using our loglikelihood values 
neg_doc_final = get_scaled_list(top_neg_words)
neu_doc_final = get_scaled_list(top_neu_words)
pos_doc_final = get_scaled_list(top_pos_words)



# visualise our synthetic corpus
generate_word_clouds(neg_doc_final, neu_doc_final, pos_doc_final, "oxford-smart")