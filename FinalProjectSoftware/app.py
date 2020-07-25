from flask import Flask, render_template, url_for, request

import requests
import math
import ktrain
import numpy as np
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import f1_score
#from sklearn.metrics import accuracy_score
from ktrain import text
from tensorflow import keras
#import pickle
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"]="0";
import sys
#from nltk.tokenize import sent_tokenize
import spacy
import pytextrank
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer




#y_pred = predictor.predict("India must join hands with Central Asia for peace in Afghanistan")


#print(y_pred)


app = Flask(__name__)

modelno = int(sys.argv[1])

#modelno 1 -> NB-SVM
#modelno 2 -> bert language model
#modelno 3 -> mLSTM
#modelno 4 -> mLSTM (originally trained)

#Load Model

predictor = None

if modelno == 1:
	predictor = ktrain.load_predictor('models/predictorNBSVM')
elif modelno == 2:
	predictor = ktrain.load_predictor('models/predictorDistilBert')
elif modelno == 3:
	predictor = ktrain.load_predictor('models/predictorMLSTM')
else:
	pass



def color(score, pred):
	# rgb(red, green, blue)
	red = 0
	green = 0
	blue = 0
	if pred == '0': #red
		#https://stackoverflow.com/questions/5731863/mapping-a-numeric-range-onto-another
		slope = -(255.0 - 120.0) / (1.0 - 0.05)
		output = 255.0 + slope * (score - 0.05)
		red = int(output)
	elif pred == '1': #neutral
		slope = -(255.0 - 120.0) / (0.05 - (-0.05))
		output = 255.0 + slope * (score - (-0.05))
		blue = int(output)
	else:
		slope = -(255.0 - 120.0) / (1.0 - 0.05)
		output = 255.0 + slope * (score - 0.05)
		green = int(output)

	return str((red, green, blue))
	

def getPrediction(chunk):
	#return predictor.predict(chunk)
	sid = SentimentIntensityAnalyzer()
	scores = sid.polarity_scores(chunk)
	#print(scores)
	if modelno == 1 or modelno ==2 or modelno == 3:
		return predictor.predict(chunk), abs(scores['compound'])

	if scores['compound'] >= 0.05:
		return '2', abs(scores['compound'])
	elif scores['compound'] >= -0.05:
		return '1', abs(scores['compound'])
	else:
		return '0', abs(scores['compound'])
 
#text to sentences, simple !!!
def breakIntoSentences(incoming):
	incoming = incoming.split('.')
	newincoming = []
	for i in incoming:
		newincoming.append(i+'.')
	return newincoming
	#return sent_tokenize(incoming);

def breakIntoWordsAndPhrases(text):
	nlp = spacy.load("en_core_web_sm")
	tr = pytextrank.TextRank()
	nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
	doc = nlp(text)
	hashset = set()
	for p in doc._.phrases:
		for q in p.chunks:
			hashset.add(str(q))

	indextophrases = {}
	for s in hashset:
		indextophrases[text.find(s)] = s

	i = 0
	end = len(text)
	chunks = []
	string = ""
	while i < end:
		if i in indextophrases:
			chunks.append(string)
			chunks.append(indextophrases[i])
			i += len(indextophrases[i])
			string = ""
		else:
			string += text[i]
			i += 1
			if i==end: chunks.append(string)

	return chunks

#text to chunks, based on words, phrases and zero shot classifier
def breakIntoChunks(text):

	chunks = breakIntoWordsAndPhrases(text)

	#now our chunks are ready based on words and phrases
	#now we would be breaking into meaningful chunks using zeroshotclassifier

	meaningfulchunks = []
	
	i = 0
	end = len(chunks)
	labelprev = -1
	labelcurr = -1
	stringprev = ""
	stringcurr = ""
	score = 0.0
	prevscore = 0.0
	while i< end:
		if i==0:
			
			labelcurr, score = getPrediction(chunks[i])#predictor.predict(chunks[i])
			stringcurr = chunks[i] 
		else:
			labelcurr, score = getPrediction(stringprev + chunks[i])#predictor.predict(chunks[i])
			stringcurr = stringprev + chunks[i]
			if labelcurr != labelprev or score < prevscore:
				meaningfulchunks.append(stringprev)
				stringcurr = chunks[i]
				#labelcurr = predictor.predict(chunks[i])		
		
		labelprev = labelcurr
		stringprev = stringcurr
		prevscore = score
		i += 1
		if i == end: meaningfulchunks.append(stringprev)

	return meaningfulchunks
			

def buildTextToPrint(sentences):
	
	texttoprint = ''
	for sentence in sentences:
		ypred, score = getPrediction(sentence) #predictor.predict(sentence)
		#print(ypred, score)
		texttoprint += '<font style = "font-family: Times New Roman; text-align: justify; font-size:25px; color: rgb'
		if ypred == '0': texttoprint = texttoprint + color(score, ypred)+ '">' + sentence + '</font>'
		elif ypred == '1': texttoprint = texttoprint + color(score, ypred)+ '">' + sentence + '</font>'
		else: texttoprint = texttoprint + color(score, ypred) + '">' + sentence + '</font>'

	return texttoprint


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/contact')
def contact():
	return render_template('contact.html')

@app.route('/readEnglish', methods=['GET', 'POST'])
def readEnglish():
	return render_template('readEnglish.html')

@app.route('/readUrdu', methods=['GET', 'POST'])
def readUrdu():
	return render_template('readUrdu.html')


@app.route('/menuEnglish', methods=['GET', 'POST'])
def menuEnglish():
	return render_template('menuEnglish.html')

@app.route('/showExample', methods=['GET', 'POST'])
def showExample():
	incoming = request.form['showExampleNumber'];
	incoming = incoming.split(' ');
	text = ""
	texttoprint = ""
	fname = 'news/' + incoming[0] + '/'+ incoming[1]+'.txt'
	#f = open(fname, "r")
	#text = bytes(f.read(), 'utf-8').decode('utf-8', 'ignore')
	with open(fname, encoding="latin-1") as ff:
		text = ff.read()
		sentences = breakIntoChunks(text)
	
		texttoprint = '<div id="anim">'
		texttoprint += buildTextToPrint(sentences)
		texttoprint += '</div>'
	return render_template('showEnglish.html', pred=texttoprint)

@app.route('/exampleEnglish', methods=['GET', 'POST'])
def exampleEnglish():
	incoming = request.form['exampleEnglish'];

	if(incoming == 'Demo'):
		return render_template('demo.html')
	if(incoming == 'Army'):
		return render_template('army.html')
	if(incoming == 'Isro'):
		return render_template('isro.html')
	if(incoming == 'India'):
		return render_template('india.html')
	if(incoming == 'Drdo'):
		return render_template('drdo.html')

	return render_template('kashmir.html')

@app.route('/exampleUrdu', methods=['GET', 'POST'])
def exampleUrdu():
	return render_template('exampleUrdu.html')


@app.route('/goLive', methods=['GET', 'POST'])
def goLive():
	return render_template('goLive.html')


@app.route('/goLiveShow', methods=['GET', 'POST'])
def goLiveShow():
	incoming = request.form['keyword'];
	query = 'q='
	query = query + incoming + '&'
	url = ('http://newsapi.org/v2/everything?'+
	query+
       'from=2020-06-25&'
       'sortBy=popularity&'
       'apiKey=9b101bf919c24b0a8aea24a66ab1e1fc')
	response = requests.get(url)
	length = len(response.json()['articles'])
	string = '<div id="anim">'
	for i in range(min(10, length)):
		string += buildTextToPrint(breakIntoChunks(response.json()['articles'][i]['title']))
		string += '<br/>'
	string += '</div>'
	return render_template('goLiveShow.html', data = string)

@app.route('/showEnglish', methods=['GET', 'POST'])
def showEnglish():
	incoming = request.form['newsArticle'];
	dropdown = request.form['dropdown'];
	incoming = incoming.strip()
	sentences = []
	if dropdown == "-1" or dropdown == "3":
		sentences = breakIntoChunks(incoming)
	elif dropdown == "1":
		sentences = breakIntoSentences(incoming)
	elif dropdown == "2":
		sentences = breakIntoWordsAndPhrases(incoming)

	texttoprint = '<div id="anim">'
	texttoprint += buildTextToPrint(sentences)
	texttoprint += '</div>'
	#print(texttoprint)
	return render_template('showEnglish.html', pred=texttoprint)

@app.route('/showUrdu', methods=['GET', 'POST'])
def showUrdu():
	urdutext = request.form['urduArticle'];
	urdutext = urdutext.strip()
	urdutext = [st.strip() for st in urdutext.splitlines()]
	urdutext = ' '.join(urdutext)
	headers = {'Content-Type': 'application/json; charset=utf-8',}
	params = (('version', '2018-05-01'),)
	data = '{"text": ["'+urdutext+'"], "model_id":"ur-en"}'
	response = requests.post('https://api.eu-gb.language-translator.watson.cloud.ibm.com/instances/33ae3fe4-df2e-4769-a2da-9b6f6433946f/v3/translate?version=2018-05-01', headers=headers, data=data.encode('utf-8'), auth=('apikey', 'Ygir-J0aZEpK6fava68HuLjpwpVPAUVycztQzfsPtP-N'))

	if 'translations' in response.json().keys():
		incoming = response.json()['translations'][0]['translation']
		incoming = incoming.rstrip()
		incoming = incoming.lstrip()
		sentences = breakIntoChunks(incoming)
		texttoprint = '<div id="anim">'
		texttoprint += buildTextToPrint(sentences)
		texttoprint += '</div>'
		#print(urdutext+"<br/><br/>"+texttoprint)
		return render_template('showUrdu.html', pred=urdutext+"<br/><br/>"+texttoprint)
	else:
		return render_template('showUrdu.html', pred=urdutext+"<br/><br/>"+"Invalid input")

if __name__=="__main__":
	app.run()
