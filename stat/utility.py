

import re, tweepy, datetime, time, csv
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns









def get_tweets_raw(filename):
	db_result = pd.DataFrame(columns = ['id', 'clean_tweet', 'subjectivity', 'polarity', 'sentiment'])
	# Advanced CSV loading example
	filename_in=filename+'.csv'
	#print(filename_in)
	df = pd.read_csv(filename_in,usecols=['id', 'location'])
	print(df['location'].value_counts(ascending=True))

	target_cnt = Counter(df.location)

	plt.figure(figsize=(16,8))
	plt.bar(target_cnt.keys(), target_cnt.values())
	plt.title("Dataset labels distribuition")


	#df = pd.read_csv(filename_in)
	#print(df.head())
	# print('Total no. of tweets is {}'.format(len(df)))
	# df = df.drop_duplicates(subset = ["text"])
	# print('Total no. of unique tweets is {}'.format(len(df)))
	

	# for x in range (len(df)):
	# 	#print(df.iloc[i, 0], df.iloc[i, 2])
	# 	id_t=df.iloc[x, 0]
	# 	tweet=df.iloc[x, 1]
	# 	tweet=str(tweet).lower()
	# 	clean_text=clean_tweet(tweet)
		
	# 	subjectivity, polarity, sentiment = get_tweet_sentiment(clean_text)
	# 	ith_row = [id_t, clean_text, subjectivity,polarity, sentiment]
	# 	# Append to dataframe - db_tweets
	# 	db_result.loc[len(db_result)] = ith_row
	
	
	# db_result = db_result.drop_duplicates(subset = ["clean_tweet"])
	# print('Total no. of unique cleaned tweets is {}'.format(len(db_result)))

	# filename_out=filename+'_result.csv'	
	# db_result.to_csv(filename_out, index = False)
	# sentiment_df=db_result["polarity"]
	# sub_df=db_result["subjectivity"]
	# plot_histo_polarity(filename,sentiment_df)
	# plot_histo_subjectivity(filename,sub_df)



