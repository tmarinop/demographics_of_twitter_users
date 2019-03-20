import pymongo
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pprint import pprint
import sys
import pylab
import csv
from datetime import datetime, date
import dateutil.parser as dparser
from collections import Counter
import collections 
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from pymongo import MongoClient
from string import punctuation
from collections import Counter
import re
import string
import emoji
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
import statistics
from spherecluster import SphericalKMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc






df = pd.read_excel('ground-truth-age.xlsx', sheet_name='Valid')
#listId = df[['User_id','Age']]
#listId = [unicode(item) for item in listId]
#listage = df['Age']

df2 = pd.read_csv('ground_truth_tfidf.csv')
df_tr = df2
ages = df_tr[['ID', 'Age']]

connection = MongoClient('localhost', 27017)
db = connection.faloutsos
result = db.NYC_Network.find()
tx=[]
word_list=[]
fdic={}
tw={}
cr_date={}


emoticons_str = r"""
	(?:
		[:=;] # Eyes
		[oO\-]? # Nose (optional)
		[D\)\]\(\]/\\OpP] # Mouth
	)"""
 
regex_str = [
	emoticons_str,
	r'<[^>]+>', # HTML tags
	r'(?:@[\w_]+)', # @-mentions
	r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
	r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
	r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
	r'(?:[\w_]+)', # other words
	r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
	return tokens_re.findall(s)

def preprocess(s, lowercase=False):
	tokens = tokenize(s)
	if lowercase:
		tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
	return tokens

def extract_emojis(text):
 return [c for c in text if c in emoji.UNICODE_EMOJI]

def count_emoji(text):
 extract = extract_emojis(text)
 emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
 emojilist = emoji_pattern.findall(text)
 emojilist = [filter(None,emoji) for emoji in emojilist]
 return emojilist+extract

slang_words=[]
with open('slang.txt') as file:
   for row in file:
   	slang_words.append(row.split()[0])

def countSlang(tokens):
	slangCounter = 0
	slangsFound = []
	for word in tokens:
		if word in slang_words:
			slangsFound.append(word)
			slangCounter += 1
	return slangCounter, slangsFound

def countAllCaps(text):
    """ Input: a text, Output: how many words are all caps """
    return len(re.findall("[A-Z0-9]{3,}", text))

#def char_is_emoji(character):
#    return character in emoji.UNICODE_EMOJI

#def text_has_emoji(text):
#    for character in text:
#        if character in emoji.UNICODE_EMOJI:
#            return character
  
def process(text, tokenizer=TweetTokenizer(), stopwords=[]):
    """Process the text of a tweet:
    - Lowercase
    - Tokenize
    - Stopword removal
    - Digits removal
    Return: list of strings
    """
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    # If we want to normalize contraction, uncomment this
    # tokens = normalize_contractions(tokens)
    return [tok for tok in tokens if tok not in stopwords and not tok.isdigit()]

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text


finalpunc = []
finalids = []
finalsents =[]
finalcaps = []
finalslangs = []
finalemojis = []
finaltopterms = []
finaltophashtags = []
finalhashtags = []
finalfriends = []
finalfollowers = []
finalavgtweets = [] 
finalslangwords = [] 
finalage = []
finalavgwordlength = []
finalmentions = []
alltweets = []
termsperuser = []
vectors = []
i=0

for doc in result[0:6070]:
	for ids in df['User_id'].astype(unicode):
		if ids == doc['user_id']:
			i+=1
			print '# of users found:', i 
			print ids
			print df[df['User_id'].astype(unicode)==unicode(ids)]['Age']
			 

			finalage.append(df[df['User_id'].astype(unicode)==ids]['Age'].item())
			finalids.append(ids)

			friends = str(doc['friends_count'])
			finalfriends.append(friends)
			#print "friends_count:", friends

			followers = str(doc['followers_count'])
			finalfollowers.append(followers)
			#print "followers_count: ", followers

			tweets = doc['statuses_count']
			account_created_date = doc['created_at']
			delta = datetime.utcnow() - account_created_date
			account_age_days = delta.days
			#print("Account age (in days): " + str(account_age_days))
			if account_age_days > 0:
				avgtweets=("Average tweets per day: " + "%.2f"%(float(tweets)/float(account_age_days)))
				finalavgtweets.append("%.2f"%(float(tweets)/float(account_age_days)))
				#print avgtweets
				punc_count = []
				word_count = []
				count_all = Counter()
				sum_of_tweets=0
				emoji_count=[]
				sents_count =[]
				totalSlangs = 0
				totalSlangsFound = []
				totalAllCaps = 0
				tweet_tokenizer = TweetTokenizer()
				terms = []
				terms_hash = []
				terms_mention = []
				sumemoji = []
				hashtagsFound = []
				for tweet in doc['tweets']:
					#print tweet['text']
					sum_of_tweets+=1
					#print sum_of_tweets
					if not tweet['retweeted'] and 'RT @' not in tweet['text']:

					#remove urls
						tweet['text'] = re.sub(r'http\S+', 'url', tweet['text'])

					#count words
						word_count.append(len(tweet['text'].split()))

					#count punctuation
						punc_count.append(len([char for char in tweet['text'] if char in string.punctuation]))
					
					#count emoji
						sumemoji.append(len(count_emoji(tweet['text'])))

					#count sentence length
						sents_count.append(tweet['text'].split('.'))
						avg_len = sum(len(tweet['text'].split()) for x in sents_count) / len(sents_count)

					#count words with all caps
						totalAllCaps += countAllCaps(tweet['text'])

					#average word length
						words = tweet['text'].split()
						avg_word_length = sum(len(word) for word in words) / len(words)


					#tokenization
					#tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
					#emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
					#print(preprocess(tweet['text']))
					#create a list of all terms
					#update the counter
					#terms_all = [term for term in preprocess(tweet['text'])]
					#count_all.update(terms_all)
					#print terms_all

					#remove punctuation and stopwords
						punctuation = list(string.punctuation)
						stop = stopwords.words('english') + punctuation + ['rt', 'via']
						tokens = process(text=removeUnicode(tweet['text']), tokenizer=tweet_tokenizer, stopwords=stop)
					#count_all.update(tokens)

					
					#count slang words
						temp_slangs, temp_slangsFound = countSlang(tokens)
						totalSlangs += temp_slangs
						for word in temp_slangsFound:
							totalSlangsFound.append(word)


					#count terms only (no hashtags, no mentions)
						terms_only = " ".join([term for term in process(text=removeUnicode(tweet['text']), tokenizer=tweet_tokenizer, stopwords=stop) if term not in stop and not term.startswith(('#', '@'))]) 	
					#count_all.update(terms_only)
					#print terms_only
						terms.append(terms_only)
						alltweets.append(terms_only)

					
					#count hashtags only
						terms_hash.append(len([term for term in process(text=removeUnicode(tweet['text']), tokenizer=tweet_tokenizer, stopwords=stop) if term.startswith('#')]))
						hashtags = " ".join([term for term in process(text=removeUnicode(tweet['text']), tokenizer=tweet_tokenizer, stopwords=stop) if term.startswith('#')])
						hashtagsFound.append(hashtags)

					#count_all.update(terms_hash)

					#count mentions only
						terms_mention.append(len([term for term in process(text=removeUnicode(tweet['text']), tokenizer=tweet_tokenizer, stopwords=stop) if term.startswith('@')]))

				'''
				print 'len words', len(words)
				print "Avg mentions/user: ", str(round(sum(terms_mention)/float(sum(word_count))*100,2))
				print "Avg hashtags/user: ", str(round(sum(terms_hash)/float(sum(word_count))*100, 2))
				print "Avg word length/user: ", str(round(avg_word_length,2))
				print "Avg words/user: ", str(round(sum(word_count)/float(sum_of_tweets),2))
				print "Avg punctuation/user: ", str(round(sum(punc_count)/float(sum_of_tweets),2))
				print "Sum emojis/user: ", sum(sumemoji)
				print "Avg sentence length/user: ", str(round(avg_len,2))
				print "Avg capitalized words: ", str(round((totalAllCaps/float(sum(word_count)))*100,2))
				print "Sum slang words/user: ", totalSlangs
				print "Avg hashtags/user: ", str(round(sum(terms_hash)/float(sum(word_count))*100,2))
				print "Avg mentions/user: ", str(round(sum(terms_mention)/float(sum(word_count))*100,2))
				print list(set(totalSlangsFound))
				'''
				termsperuser.append(terms)
				#print termsperuser
				#print(count_all.most_common(5))

				#print doc['tweets'][11]['text']
				#print [' '.join(c for c in doc['tweets'][389]['text'])]
				#print sumemoji[11]
				#print punc_count[11]

				hashtagsFound = filter(None,hashtagsFound)

				finalavgwordlength.append(str(round(avg_word_length,2)))
				finalpunc.append(str(round(sum(punc_count)/float(sum_of_tweets),2)))
				finalsents.append(str(round(avg_len,2)))
				finalcaps.append(str(round((totalAllCaps/float(sum(word_count)))*100,2)))
				finalslangs.append(totalSlangs)
				finalemojis.append(sum(sumemoji))
				finalhashtags.append(str(round(sum(terms_hash)/float(sum(word_count))*100,2)))
				finalmentions.append(str(round(sum(terms_mention)/float(sum(word_count))*100,2)))
				finalslangwords.append(list(set(totalSlangsFound)))

				#finaltopterms.append(count_all.most_common(5))
				#finalhashtags.append(count_all.most_common(5))


user_texts_concat = []

vectorizer = TfidfVectorizer()


vectorizer.fit(alltweets)
#print len(termsperuser)
#print len(termsperuser[0])
#print type(termsperuser[0][0])
for user_text in termsperuser:
	total_text = ""
	for text in user_text:
		total_text += text
	user_texts_concat.append(total_text)

kminput = vectorizer.transform(user_texts_concat)
hier_input = kminput.toarray()

#print len(vectorizer.vocabulary_)
#print len(alltweets)
features = vectorizer.get_feature_names()
indices = np.argsort(vectorizer.idf_)[::-1]
top_n = 5
top_features = [features[y] for y in indices[:top_n]]
#print top_features	
#dend = shc.dendrogram(shc.linkage(hier_input, method='ward'))
#plt.show()

#Clustering tfidf (edit clustering algorithm and range of clusters)
range_n_clusters = [4]
for n_clusters in range_n_clusters:
	kmeans = AgglomerativeClustering(n_clusters=n_clusters).fit(hier_input)
	silhouette_avg = silhouette_score(hier_input, kmeans.labels_)
	print silhouette_avg
	ages['Clusters'] = kmeans.labels_
	print ages.groupby(['Clusters'])['Age'].mean()
	print "Std", ages.groupby(['Clusters'])['Age'].std()
	x=list(ages.groupby(['Clusters'])['Age'])
	for t in x:
		mydic = {}
		for age in t[1]:
			if age not in mydic:
				mydic[age]=0
			mydic[age]+=1
		x=[]
		y=[]
		for age in mydic:
			x.append(age)
			y.append(mydic[age])
		#plt.hist(t[1],label="Cluster {}".format(t[0]))
		print statistics.median(x)
		plt.plot(x,y,label="Cluster {}".format(t[0]))
	plt.xlabel('Age')
	plt.ylabel('Number of users')
	plt.legend()
	plt.show()

 	#model = SilhouetteVisualizer(KMeans(n_clusters), title="Silhouette score of Kmeans clustering")    
 	#model.fit(kminput) # Fit the training data to the visualizer
 	#model.poof() # Draw/show/poof the data
 	#print "For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg
	#plt.figure()
	#plt.show()




'''
kmeans = KMeans(n_clusters=5).fit(kminput)
centeroids = kmeans.cluster_centers_
ages['Clusters'] = kmeans.labels_
iterations = kmeans.n_iter_
print(ages.groupby(['Clusters'])['Age'].mean())
'''

'''
#write features per user
rows = zip(finalids, finalage, finalfriends, finalfollowers, finalavgtweets, finalavgwordlength, finalsents, finalpunc, finalcaps, finalslangs, finalemojis, finalslangwords)

with open('ground_truth.csv', mode='w') as csvFile:
	writer = csv.writer(csvFile, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_NONNUMERIC)
	writer.writerow(("ID", "Age", "Friends_count", "Followers_count", "Avg_Tweet/day", "Avg_Word_length", "Avg_Sentence_length", "Avg_Punctuation", "Avg_Capitalized_Words", "Sum_Slang_words", "Sum_emojis","Slang_words_found")) 

	for row in rows:
		writer.writerow(row)

'''






	









