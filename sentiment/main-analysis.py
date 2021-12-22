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
program_start = time.time()


get_tweets(filename,data)

# # Advanced CSV loading example
# filename_in=filename+'.csv'
# #print(filename_in)
# df = pd.read_csv(filename_in,usecols=['id', 'date', 'sentiment'])


# # Convert dates
# # df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
# plot_by_date(filename, data, df)
program_end = time.time()
print('Total time taken to plot is {} minutes.'.format(round(program_end - program_start)/60, 2))






