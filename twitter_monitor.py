"""
Script for running the twitter monitor listener/processor
"""
import argparse
import time
import tweepy
import logging
import pandas as pd

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


from scrap import *
from config import Config


#load the args & config
parser = argparse.ArgumentParser("Run the twitter monitor listener/processor")
parser.add_argument("--configfile", "-c", default="config.json", required=False, help="Path to the config file to use.")
parser.add_argument("--logfile", "-l", default="tmlog.txt", required=False, help="Path to the log file to write to.")
parser.add_argument("--sinceid", "-s", default="0", required=False, help="Collected last twitter id")
args = parser.parse_args()

config = Config.load(args.configfile)
name= args.logfile
sid=int(args.sinceid,10)


#load the credentials and initialize tweepy
#auth  = tweepy.OAuthHandler(config.api_key, config.api_secret_key)
#auth.set_access_token(config.access_token, config.access_token_secret)
auth = tweepy.AppAuthHandler(config.api_key, config.api_secret_key)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)




# Initialise these variables:
search_words = config.filter_keywords
print(search_words)
#search_words = "#hongkong OR #hkprotests OR #freehongkong OR #hongkongprotests OR #hkpolicebrutality OR #antichinazi OR #standwithhongkong OR #hkpolicestate OR #HKpoliceterrorist OR #standwithhk OR #hkpoliceterrorism"
date_since = "2021-04-13"
numTweets = 2500
numRuns = 1
# Call the function scraptweets
scraptweets(api, search_words, date_since, numTweets, numRuns, name, sid)




