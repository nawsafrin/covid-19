import argparse
from utility import *
import re, tweepy, datetime, time, csv
import pandas as pd
from textblob import TextBlob

#load the args & config
parser = argparse.ArgumentParser("Run the sentiment analysis using TextBlob")
parser.add_argument("--input", "-i", default="", required=False, help="Path to the stored tweets csv file to use without extension")

args = parser.parse_args()

filename=args.input
program_start = time.time()
get_tweets(filename)
program_end = time.time()
print('Total time taken to analyze sentiment is {} minutes.'.format(round(program_end - program_start)/60, 2))
