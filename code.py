import numpy as np 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import sent_tokenize
#import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def readTxt(path):
	f = open(path,'r').read()
	sentence = sent_tokenize(f)
	return [s.replace('\n', '') for s in sentence if s]

def featurize(sent, word_tokenizer, top_word, all_token, freq_word, pos_list):
	rtn = []
	token = nltk.word_tokenize(sent.lower())
	words = word_tokenizer.tokenize(sent.lower())
	vocab = set(words)

	num_word = len(sent)
	div_word = len(vocab)/float(len(words)) if len(words) != 0 else 0
	num_comma = token.count(',')
	num_semi = token.count(';')
	num_col = token.count(':')
	num_dig = sum(t.isdigit() for t in token)
	rtn = [num_word, div_word, num_comma, num_semi, num_col, num_dig]

	for j in range(top_word):
		rtn.append(token.count(freq_word[j]))

	tmp = []
	pos = nltk.pos_tag(token)
	for p in pos:
		tmp.append(p[1])
	for j in range(len(pos_list)):
		rtn.append(tmp.count(pos_list[j]))
	return rtn

def main():
	author = ['Jean Paul Marat', 'William Skeen', 'Thomas Hunt Morgan', 'Chas. H. Brown', 'James Tod','Russell A. Kelly', 'Augustus Le Plongeon', 'Kabir', 'Battiscombe G. Gunn', 'Jacob Kainen']
	pathes = ['corpus1.txt','corpus2.txt', 'corpus3.txt', 'corpus4.txt', 'corpus5.txt', 'corpus6.txt', 'corpus7.txt', 'corpus8.txt', 'corpus9.txt', 'corpus0.txt']
	sent_x, data_y, data_x = [], [], []
	for i, path in enumerate(pathes):
		sent = readTxt(path)
		sent_x.append(sent)
		data_y.append([author[i]] * len(sent))
		print len(sent)
	sent_x = list(np.concatenate(sent_x))
	data_y = list(np.concatenate(data_y))
	all_sent = ' '.join(sent_x)
	
	word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	top_word = 10
	all_token = nltk.word_tokenize(all_sent.lower())
	freq = nltk.FreqDist(all_token)
	freq_word = freq.keys()[:top_word]
	pos_list = ['NN', 'NNP', 'DT', 'IN', 'JJ', 'NNS']

	print len(data_y), len(sent_x)
	for sent in sent_x:
		data_x.append(featurize(sent, word_tokenizer, top_word, all_token, freq_word, pos_list))

	le = preprocessing.LabelEncoder()
	le.fit(data_y)
	data_y = le.transform(data_y)
	
	X, Y = shuffle(data_x, data_y, random_state = 0)
	train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

	clf = RandomForestClassifier(max_depth = 8, random_state = 0)
	clf.fit(train_x, train_y)
	pred = clf.predict(test_x)
	print 'RF', accuracy_score(test_y, pred)

	clf2 = LogisticRegression(penalty = 'l1')
	clf2.fit(train_x, train_y)	
	pred = clf2.predict(test_x)
	print 'LR', accuracy_score(test_y, pred)
 


main()