import argparse
from utility import *
import re, tweepy, datetime, time, csv
import pandas as pd
from textblob import TextBlob

#load the args & config
parser = argparse.ArgumentParser("Run the sentiment analysis using TextBlob")
parser.add_argument("--input", "-i", default="", required=True, help="Path to the stored tweets csv file to use without extension")
parser.add_argument("--name", "-n", default="", required=True, help="Data Name")


args = parser.parse_args()

filename=args.input
data=args.name
filename_out=filename+'_result.csv'	
db_result = pd.read_csv(filename_out,usecols=['polarity', 'subjectivity'])
sentiment_df=db_result["polarity"]
sub_df=db_result["subjectivity"]
filename_n=filename+'_new_'
program_start = time.time()
plot_histo_polarity(filename_n,data,sentiment_df)
plot_histo_subjectivity(filename_n,data,sub_df)
#get_tweets(filename,data)
program_end = time.time()
print('Total time taken to analyze sentiment is {} minutes.'.format(round(program_end - program_start)/60, 2))
