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
import re
from datetime import datetime, date
import dateutil.parser as dparser
from collections import Counter
import collections 

#Connection & print of MongoDB Documents
from pymongo import MongoClient
connection = MongoClient('localhost', 27017)
db = connection.faloutsos
result = db.NYC_Network.find()
tx=[]
word_list=[]
fdic={}
tw={}
cr_date={}
for doc in result[0:1000]:
	#print(doc['user_id'])
	#str=['husband', 'birthday', 'bday', 'Birthday', 'Bday', 'daughter', 'years', 'old', 'year']
	str=['thirty','fourteen', 'fifteen','Fifteen', 'birthday', 'Birthday', 'college','son', 'husband', 'wife', 'Bday', 'bday', 'year', 'Year', 'years', 'mother', 'boyfriend', 'gift', 'eleven', 'seventeen', 'eighteen', 'tweenty', 'nineteen', 'sixteen', 'Fourteen', 'old', 'wishes', 'sister', 'brother','family', 'pension', 'salary', 'married', 'celebrat', 'gift', 'daughter', 'granddaughter','parents','dad', 'born', 'turned', 'school', 'work']
	count=0
	#if (doc['user_id']=="735883722587820032"):
	for x in doc['tweets']:
		s=x['text']
		#print(s)
		for item in str:
			if item in s:
				print("success")
				x_id=x['id']
				tw[x_id]={}
				#print(tw)
				tw[x_id][s]={}
				#print(tw)
				tw[x_id][s]=x['created_at'].date()
				#print(tw)
			else:
				print("fail")
	print(tw)
	print(doc['user_id'])
#dic me user_id:tweet_id:extracted text:tweet_date
	fdic[doc['user_id']]=tw
for el in fdic:
	print(el)
	print(fdic.values())
	print(' ')
print(len(fdic))