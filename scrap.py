
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import json
import pandas as pd
import csv
import re
from textblob import TextBlob
import string
import preprocessor as p
import os
import time

def scraptweets(api, search_words, date_since, numTweets, numRuns, name, sid):
    
    # Define a for-loop to generate tweets at regular intervals
    # We cannot make large API call in one go. Hence, let's try T times
    max_per_api=100
    # Define a pandas dataframe to store the date:
    db_tweets = pd.DataFrame(columns = ['id', 'username', 'acctdesc', 'location', 'following',
                                        'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts',
                                        'retweetcount', 'text', 'hashtags']
                                )
    db_tweetid = pd.DataFrame(columns = ['id'])
    program_start = time.time()
    for i in range(0, numRuns):
        # We will time how long it takes to scrape tweets for each run:
        start_run = time.time()
        
        # Collect tweets using the Cursor object
        # .Cursor() returns an object that you can iterate or loop over to access the data collected.
        # Each item in the iterator has various attributes that you can access to get information about each tweet
#result_type="recent",
#numTweets
        tweets = tweepy.Cursor(api.search, q=search_words, lang="en", count=max_per_api, tweet_mode='extended', since_id=sid).items()
		# Store these tweets into a python list
        tweet_list = [tweet for tweet in tweets]
		# Obtain the following info (methods to call them out):
        # user.screen_name - twitter handle
        # user.description - description of account
        # user.location - where is he tweeting from
        # user.friends_count - no. of other users that user is following (following)
        # user.followers_count - no. of other users who are following this user (followers)
        # user.statuses_count - total tweets by user
        # user.created_at - when the user account was created
        # created_at - when the tweet was created
        # retweet_count - no. of retweets
        # (deprecated) user.favourites_count - probably total no. of tweets that is favourited by user
        # retweeted_status.full_text - full text of the tweet
        # tweet.entities['hashtags'] - hashtags in the tweet
		# Begin scraping the tweets individually:
        noTweets = 0
        for tweet in tweet_list:
		#unique twitter id
            tid=tweet._json['id']
		# Pull the values
            username = tweet.user.screen_name
            acctdesc = tweet.user.description
            location = tweet.user.location
            following = tweet.user.friends_count
            followers = tweet.user.followers_count
            totaltweets = tweet.user.statuses_count
            usercreatedts = tweet.user.created_at
            tweetcreatedts = tweet.created_at
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                text = tweet.full_text
			# Add the 11 variables to the empty list - ith_tweet:
            ith_tweet = [tid, username, acctdesc, location, following, followers, totaltweets, usercreatedts, tweetcreatedts, retweetcount, text, hashtags]
			# Append to dataframe - db_tweets
            db_tweets.loc[len(db_tweets)] = ith_tweet
            db_tweetid.loc[len(db_tweetid)] = [tid]
		    # increase counter - noTweets  
            noTweets += 1
        
        # Run ended:
        end_run = time.time()
        duration_run = round((end_run-start_run)/60, 2)
        
        print('no. of tweets scraped for run {} is {}'.format(i + 1, noTweets))
        print('time take for {} run to complete is {} mins'.format(i+1, duration_run))
        
        #time.sleep(920) #15 minute sleep time
	


	# Once all runs have completed, save them to a single csv file:
    from datetime import datetime
		
	# Obtain timestamp in a readable format
    to_csv_timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
	# Define working path and filename
    path = os.getcwd()
    filename = path + '/data/' + to_csv_timestamp + name + '_tweets.csv'
    filename2 = path + '/data/' + to_csv_timestamp + name + '_tweet_id.csv'
	# Store dataframe in csv with creation date timestamp
    db_tweets.to_csv(filename, index = False)
    db_tweetid.to_csv(filename2, index = False)
    program_end = time.time()
    print('Scraping has completed!')
    print('Total time taken to scrap is {} minutes.'.format(round(program_end - program_start)/60, 2))
