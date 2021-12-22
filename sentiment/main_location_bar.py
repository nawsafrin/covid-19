import argparse
from utility import *
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


# get_tweets(filename,data)

# Advanced CSV loading example
# filename_in=filename+'.csv'
# #print(filename_in)
# df = pd.read_csv(filename_in,usecols=['id', 'location'])

# # Convert empty string to NaN values
# nan_value = float("NaN")
# df.replace("", nan_value, inplace=True)

# df.dropna(subset = ["location"], inplace=True)
# locations = [get_location_country(x)for x in df['location']]
# df['country_all']=locations
# df['country_all']=df['country_all'].astype(str)
# str1 = " " 
# # string_list = [ (str1.join(x)) for x in df['country_all']]
# country= [ first_country(x) for x in df['country_all'] ]
# df['country']=country
# df['country'].replace("NaN", "Unknown", inplace=True)
# # df[df.country != 'Unknown']
# df['country']=[x.upper()for x in df['country'] ]
# # df.dropna(subset = ["country"], inplace=True)




filename_out=filename+'_country.csv'	
df = pd.read_csv(filename_out,usecols=['id', 'location','country'])
dfout = df['country'].value_counts().reset_index().head(10)

# df.to_csv(filename_out, index = False)

ax=df['country'].value_counts().head(10).plot.pie(labeldistance=None, autopct=autopct,legend = True)
ax.axes.get_yaxis().set_visible(False)
plt.legend(loc='center left', bbox_to_anchor=(-0.40, 0.6))

# ax=dfout.plot.bar(legend = True)
# ax.axes.get_yaxis().set_visible(False)
# plt.legend(loc='center left', bbox_to_anchor=(-0.35, 0.6))




ax.figure.savefig(filename+'_pie.pdf')
# plot.show()

# timeline = df.groupby(['country']).agg(**{'tweets': ('id', 'count')}).reset_index()
# # df['country'].hist()
# plot = df.plot.pie(y='count', figsize=(5, 5))
# plot.show()
# df.groupby(['country']).sum().plot(kind='pie', y='')


str="[['India', 'United States', 'Canada']]"
str2="[[]]"
#str=clean_tweet(str)

# Convert dates
# df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
# plot_by_date(filename, data, df)
program_end = time.time()
print('Total time taken to execute is {} minutes.'.format(round(program_end - program_start)/60, 2))






