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
dic={}
user_fofo={}
prof_count=0
for doc in result[0:1000]:
	user_id=doc['user_id']
	dic[user_id]={}
	#print(dic)
	di2={}
	
	friends=doc['friends_count']
	followers=doc['followers_count']
	count=0
	if (friends != 0):
		fofo=int(followers/friends)
	rep_score=(followers/(friends+followers))
	print(fofo)
	print(rep_score)
	print(followers)
	if (fofo>100 or rep_score==1 or followers>5000):
		prof_count=prof_count+1
	print(prof_count)
	else:	
		for x in doc['tweets']:
		#dictionary me user_id{tweet_id:date_created}
			tw_id=x['id']
			dic[user_id][tw_id] = {} 
			#print(dic)
			s=x['text']
			#print(s, "\n")
			a = re.search(r'.*(\d{2}[\/ ](\d{2}|January|Jan|February|Feb|March|Mar|April|Apr|May|May|June|Jun|July|Jul|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)[\/ ]\d{2,4}).*',s,re.IGNORECASE)
			b = re.search(r'.*(I|He|She) (is|am) ([0-9]{2}).*',s,re.IGNORECASE)
			c = re.search(r'.*(I|He|She) (is|am) in (my|his|her) (late|mid|early)? ?(tens|twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties|hundreds).*',s,re.IGNORECASE)
			d = re.search(r'.*(I|He|She) (is|am) (twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen) ?(one|two|three|four|five|six|seven|eight|nine)?.*',s,re.IGNORECASE)
			e = re.search(r'.*(age|is|@|was) ([0-9]{2}).*',s,re.IGNORECASE)
			f = re.search(r'.*(age|is|@|was) (twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen) ?(one|two|three|four|five|six|seven|eight|nine)?.*',s,re.IGNORECASE)
			g = re.search(r'.*([0-9]{2}) (yrs|years).*',s,re.IGNORECASE)
			h = re.search(r'.*(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen) ?(one|two|three|four|five|six|seven|eight|nine)? (yrs|years).*',s,re.IGNORECASE)
			i= re.search(r'.*(happybirthday|birthday|Birthday|BDay|Bday|cake|balloon|gift|present|celebration|wishes|wish).*', s, re.IGNORECASE)
			j= re.search(r'.*(mom|mother|father|girlfriend|boyfriend|teen|teenager).*', s, re.IGNORECASE)
			k= re.search(r'.*(school|work|homework|teacher|teen|teenager).*', s, re.IGNORECASE)
			l= re.search(r'.*(wife|husband|married|son|daughter).*', s, re.IGNORECASE)
			text_list=[('a',a),('b', b ),('c', c), ('d',d),('e',e),('f',f),('g',g), ('h',h),('i',i), ('j',j),('h',h),('k',k), ('l', l)]
			di1=dict(text_list)
			#print(di1)
			#print(di1.values())
			od = collections.OrderedDict(sorted(di1.items()))
			#print(od)
			for p, v in od.items():
				#print (p, v)
				dic[user_id][tw_id] = od 
	print(prof_count)
	print(dic, "\n")
	print(len(dic))
#len_dicfin=len(dic[user_id])
#print(len_dicfin)