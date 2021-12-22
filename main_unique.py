import argparse
# from utility import *
import re, tweepy, datetime, time, csv
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

#load the args & config
parser = argparse.ArgumentParser("Run the sentiment analysis using TextBlob")
parser.add_argument("--input", "-i", default="", required=True, help="Path to the stored tweets csv file to use without extension")
parser.add_argument("--name", "-n", default="", required=True, help="Data Name")


args = parser.parse_args()

filename=args.input
data=args.name
program_start = time.time()

filename_in=filename+'.csv'
#print(filename_in)
df = pd.read_csv(filename_in)
p=(len(df))
df = df.drop_duplicates(subset = ["id"])
q=(len(df))
if p!=q:
	filename_out=filename+'_unique.csv'	
	df.to_csv(filename_out, index = False)

program_end = time.time()
print('Total time taken to execute is {} minutes.'.format(round(program_end - program_start)/60, 2))