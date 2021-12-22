import re, tweepy, datetime, time, csv
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

import locationtagger

import geograpy
from geograpy import places

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import Counter

def clean_tweet(tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(tweet)).split())

#def remove_stopwords(tweet):


# This function reads a file and returns its contents as an array
def readFileandReturnAnArray(fileName, readMode, isLower):
    myArray=[]
    with open(fileName, readMode) as readHandle:
        for line in readHandle.readlines():
            lineRead = line
            if isLower:
                lineRead = lineRead.lower()
            myArray.append(lineRead.strip().lstrip())
    readHandle.close()
    return myArray

def removeItemsInTweetContainedInAList(tweet_text,stop_words,splitBy):
    wordsArray = tweet_text.split(splitBy)
    StopWords = list(set(wordsArray).intersection(set(stop_words)))
    return_str=""
    for word in wordsArray:
        if word not in StopWords:
            return_str += word + splitBy
    return return_str.strip().lstrip()




def get_stemmed_text(tweet_text):
	#tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
	# tokenized_tweet=tweet_text.split()
	# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
	stemmer = PorterStemmer()
	return ' '.join([stemmer.stem(word) for word in tweet_text.split()]) 
	
def get_lemmatized_text(tweet_text):
    
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in tweet_text.split()])





def get_tweet_sentiment(tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return analysis.sentiment.subjectivity, analysis.sentiment.polarity, 'positive'
        elif analysis.sentiment.polarity == 0:
            return analysis.sentiment.subjectivity, analysis.sentiment.polarity, 'neutral'
        else:
            return analysis.sentiment.subjectivity, analysis.sentiment.polarity, 'negative'

def get_tweets_1_raw(filename):
	db_result = pd.DataFrame(columns = ['id', 'clean_tweet', 'subjectivity', 'polarity', 'sentiment'])
	# Advanced CSV loading example
	filename_in=filename+'.csv'
	#print(filename_in)
	df = pd.read_csv(filename_in,usecols=['id', 'text'])
	#df = pd.read_csv(filename_in)
	#print(df.head())
	print('Total no. of tweets is {}'.format(len(df)))
	df = df.drop_duplicates(subset = ["text"])
	print('Total no. of unique tweets is {}'.format(len(df)))
	

	for x in range (len(df)):
		#print(df.iloc[i, 0], df.iloc[i, 2])
		id_t=df.iloc[x, 0]
		tweet=df.iloc[x, 1]
		tweet=str(tweet).lower()
		clean_text=clean_tweet(tweet)
		
		subjectivity, polarity, sentiment = get_tweet_sentiment(clean_text)
		ith_row = [id_t, clean_text, subjectivity,polarity, sentiment]
		# Append to dataframe - db_tweets
		db_result.loc[len(db_result)] = ith_row
	
	
	db_result = db_result.drop_duplicates(subset = ["clean_tweet"])
	print('Total no. of unique cleaned tweets is {}'.format(len(db_result)))

	filename_out=filename+'_result.csv'	
	db_result.to_csv(filename_out, index = False)
	sentiment_df=db_result["polarity"]
	sub_df=db_result["subjectivity"]
	plot_histo_polarity(filename,sentiment_df)
	plot_histo_subjectivity(filename,sub_df)
	

def get_tweets(filename,data):
	db_result = pd.DataFrame(columns = ['id', 'location', 'date','clean_tweet', 'subjectivity_r', 'polarity_r', 'sentiment_r','root_words', 'subjectivity_sw', 'polarity_sw', 'sentiment_sw', 'normalized_words','subjectivity', 'polarity', 'sentiment'])
	# Advanced CSV loading example
	filename_in=filename+'.csv'
	#print(filename_in)
	df = pd.read_csv(filename_in,usecols=['id', 'location', 'tweetcreatedts', 'text'])


	# Convert dates
	df['tweetcreatedts'] = pd.to_datetime(df['tweetcreatedts'], errors='coerce').dt.date



	#df = pd.read_csv(filename_in)
	#print(df.head())
	df = df.drop_duplicates(subset = ["id"])
	print('Total no. of tweets is {}'.format(len(df)))
	df = df.drop_duplicates(subset = ["text"])
	print('Total no. of unique tweets is {}'.format(len(df)))
	stop_words = readFileandReturnAnArray("stopwords_ext.txt","r",True)

	for x in range (len(df)):
		#print(df.iloc[i, 0], df.iloc[i, 2])
		id_t=df.iloc[x, 0]
		loc=df.iloc[x, 1]
		date=df.iloc[x, 2]
		tweet=df.iloc[x, 3]
		tweet=str(tweet).lower()
		clean_text=clean_tweet(tweet)
		#raw dat
		subjectivity_r, polarity_r, sentiment_r = get_tweet_sentiment(clean_text)
		clean_root_words = removeItemsInTweetContainedInAList(clean_text.strip().lstrip(),stop_words, " ")
		# removing stop words
		subjectivity_sw, polarity_sw, sentiment_sw = get_tweet_sentiment(clean_root_words)

		stem_words=get_stemmed_text(clean_root_words)
		lem_words=get_lemmatized_text(stem_words)


		#doing stemming and lemmazition 
		subjectivity, polarity, sentiment = get_tweet_sentiment(lem_words)


		ith_row = [id_t, loc, date, clean_text,subjectivity_r,polarity_r, sentiment_r, clean_root_words, subjectivity_sw,polarity_sw, sentiment_sw, lem_words, subjectivity, polarity, sentiment]
		# Append to dataframe - db_tweets
		db_result.loc[len(db_result)] = ith_row
	
	
	db_result_raw = db_result.drop_duplicates(subset = ["clean_tweet"])
	print('Total no. of unique (raw) cleaned tweets is {}'.format(len(db_result_raw)))

	db_result_sw = db_result.drop_duplicates(subset = ["root_words"])
	print('Total no. of unique cleaned tweets after stop word removal is {}'.format(len(db_result_sw)))

	db_result_n = db_result.drop_duplicates(subset = ["normalized_words"])
	print('Total no. of unique cleaned tweets after normalization is {}'.format(len(db_result_n)))

	filename_out=filename+'_result.csv'	
	db_result.to_csv(filename_out, index = False)

	# target_cnt_r = Counter(db_result.sentiment_r)
	# target_cnt_sw = Counter(db_result.sentiment_sw)
	# target_cnt_n = Counter(db_result.sentiment_n)


	# data=[[target_cnt_r['positive'],target_cnt_r['negative'], target_cnt_r['neutral']],[target_cnt_sw['positive'],target_cnt_sw['negative'], target_cnt_sw['neutral']],[target_cnt_n['positive'],target_cnt_n['negative'], target_cnt_n['neutral']]]
	# X = np.arange(3)
	# fig=plt.figure(figsize=(6,4))
	# Senti=['Raw','StopWords Removal','Normalized Words']
	# # ax = fig.add_axes([0,0,1,1])
	# # plt.figure(figsize=(6,4))
	# # plt.bar(target_cnt_r.keys(), target_cnt_r.values(),color='r')
	# # plt.bar(target_cnt_sw.keys(), target_cnt_sw.values(),color='b')
	# # plt.bar(target_cnt_n.keys(), target_cnt_n.values(),color='g' )
	# # plt.legend(labels=['Raw', 'StopWords Removal', 'Normalized Words'])
	# plt.legend(Senti,loc=2)

	# plt.bar(X + 0.00, data[0], color = 'r', width = 0.25)
	# plt.bar(X + 0.25, data[1], color = 'b', width = 0.25)
	# plt.bar(X + 0.50, data[2], color = 'g', width = 0.25)


	# plt.title("Sentiment distribuition")
	# plt.savefig(filename+'distribuition.png')


	plot_bar_comparison(filename,data,db_result)
	plot_by_date(filename,data,db_result)
	
	sentiment_df=db_result["polarity"]
	sub_df=db_result["subjectivity"]
	plot_histo_polarity(filename,data,sentiment_df)
	plot_histo_subjectivity(filename,data,sub_df)

def plot_histo_polarity(filename,data,sentiment_df):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid")
	fig, ax = plt.subplots(figsize=(10,7))

	# Plot histogram with break at zero
	sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, -0.01, 0.0, 0.01, 0.25, 0.5, 0.75, 1],
	#sentiment_df.hist(bins=[-1, -0.9, -0.8, -0.7, -0.6, 0.0, 0.01, 0.25, 0.5, 0.75, 1],
		         ax=ax,
		         color="purple")
	ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
	plt.title("Sentiments (Polarity) from Tweets on "+ data)
	#plt.show()
	plt.savefig(filename+'polarity-histo.png')

def plot_histo_subjectivity(filename,data,sentiment_df):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid")
	fig, ax = plt.subplots(figsize=(10,7))

	# Plot histogram with break at zero
	sentiment_df.hist(bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
	#sentiment_df.hist(bins=[-1, -0.9, -0.8, -0.7, -0.6, 0.0, 0.01, 0.25, 0.5, 0.75, 1],
		         ax=ax,
		         color="purple")
	ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
	plt.title("Sentiments (Subjectivity) from Tweets on " + data)
	#plt.show()
	plt.savefig(filename+'Subjectivity-histo.png')





def plot_bar_comparison(filename,name,result_df):
	Senti=['Positive','Negative','Neutral']
	label=['Raw Tweets','Tweets without StopWords','Normalized Tweets']
	target_cnt_r = Counter(result_df.sentiment_r)
	target_cnt_sw = Counter(result_df.sentiment_sw)
	target_cnt_n = Counter(result_df.sentiment)
	a=len(result_df)/100
	

	pos = np.arange(len(label))
	bar_width = 0.25
	Positive=[target_cnt_r['positive']/ a,target_cnt_sw['positive']/a,target_cnt_n['positive'] /a]
	Negative=[target_cnt_r['negative']/a,target_cnt_sw['negative']/a, target_cnt_n['negative']/a]
	Neutral=[target_cnt_r['neutral']/a,target_cnt_sw['neutral']/a, target_cnt_n['neutral']/a]

	plt.bar(pos,Positive,bar_width,color='g',edgecolor='black')
	plt.bar(pos+bar_width,Negative,bar_width,color='r',edgecolor='black')
	plt.bar(pos+bar_width+bar_width,Neutral,bar_width,color='b',edgecolor='black')

	plt.xticks(pos, label)
	plt.xlabel('Twitter Data', fontsize=16)
	plt.ylabel('Sentiment Category (%)', fontsize=16)

	plt.legend(Senti,loc=0)
	plt.title("Sentiment distribuition for " + name)
	plt.savefig(filename+'distribuition.png')

	# plt.title('Group Barchart - Happiness index across cities By Gender',fontsize=18)
	
	# plt.show()

def plot_by_date(filename, name, result_df):
	a=len(result_df)/100
	# Get counts of number of tweets by sentiment for each date
	timeline = result_df.groupby(['date', 'sentiment']).agg(**{'tweets': ('id', 'count')}).reset_index().dropna()

	# Plot results
	import plotly.express as px
	fig = px.line(timeline, x='date', y='tweets', color='sentiment', category_orders={'sentiment': ['neutral', 'negative', 'positive']},
	             title="Timeline showing sentiment of tweets about " + name)
	# fig.show()
	fig.write_image(filename+'date.pdf')


def get_location_country(loc_text):

	entities = locationtagger.find_locations(text = loc_text)
	c_list=entities.countries
	# c_list=entities.other_countries
	if not len(c_list):
		c_list.append(entities.other_countries)
	return c_list


	# if(c_list[0]):
	# 	cname=c_list[0]
	# else:
	# 	# for (x in c_list[0]):
	# 	cname=[i[0] for i in c_list]
	# return cname


	# pc = places.PlaceContext(loc_text)
	# pc.set_countries()
	# return pc.countries
	# loc_text=
	# p=geograpy.locator.Locator(db_file= loc_text, correctMisspelling=True)
	# p=geograpy.locateCity(loc_text, correctMisspelling=True)
	# return geograpy.locator.getCountry(loc_text)

	# places = geograpy.get_place_context(text = loc_text,  labels='GPE,GSP')
	# return places.countries

def split_string(str):
	return filter(None, re.split("[,\-!?:']+", str))

def first_country(str):
	co=[]
	# nan_value = float("NaN")
	x=split_string(str)
	for a in x:
		p=clean_tweet(a)
		if len(p):
			co.append(p)
			break
		# else:
		# 	co.append("NaN")

	if len(co):
		return co[0]
	else: 
		return "NaN"


def autopct(pct): # only show the label when it's > 10%
    return ('%.2f' % pct) if pct > 5 else ''