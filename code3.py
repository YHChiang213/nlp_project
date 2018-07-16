import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re, string, unicodedata
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import collections
from sklearn import preprocessing, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import pickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def readTxt(path):
	rtn = []
	f = open(path,'r').read()
	udata = f.decode("utf-8")
	main_text = udata.encode("ascii", "ignore")
	processed = re.sub("\s+", " ", main_text)
	sentences = sent_tokenize(processed.lower())

	for s in sentences:
		if not len(s) < 6:
			rtn.append(s)
	#print len(rtn)
	return rtn 

def featurize(sent, word_tokenizer, top_word, all_token, freq_word, pos_list):
	rtn = []
	token = nltk.word_tokenize(sent.lower())
	words = word_tokenizer.tokenize(sent.lower())
	p = inflect.engine()
	stemmer = LancasterStemmer()
	lemmatizer = WordNetLemmatizer()
	new_word = []
	for w in words:
		tmp2 = re.sub(r'[^\w\s]', '', w)
		if tmp2.isdigit():
			tmp3 =  p.number_to_words(tmp2)
		else:
			tmp3 = tmp2
		if tmp3 not in stopwords.words('english'):
			stem = stemmer.stem(tmp3)
			lemma = lemmatizer.lemmatize(stem, pos='v')
			new_word.append(lemma)
	vocab = set(words)

	num_word = len(sent)
	num_new_word = len(new_word)
	div_word = len(vocab)/float(len(new_word)) if len(new_word) != 0 else 0
	num_comma = token.count(',')
	num_semi = token.count(';')
	num_col = token.count(':')
	num_dig = sum(t.isdigit() for t in new_word)
	rtn = [num_word, num_new_word, div_word, num_comma, num_semi, num_col, num_dig]

	for j in range(top_word):
		rtn.append(new_word.count(freq_word[j]))

	tmp = []
	pos = nltk.pos_tag(new_word)
	for p in pos:
		tmp.append(p[1])
	for j in range(len(pos_list)):
		rtn.append(tmp.count(pos_list[j]))
	return rtn
def training(sent_train, train_y):
	count_vect = CountVectorizer()
	x_train_count = count_vect.fit_transform(sent_train)

	tfidf_tran = TfidfTransformer()
	x_train_tfidf = tfidf_tran.fit_transform(x_train_count)

	clf = MultinomialNB().fit(x_train_tfidf, train_y)
	text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB(alpha=0.01))])
	text_clf = text_clf.fit(sent_train, train_y)

	new_f_nb = text_clf.predict_proba(sent_train)

	text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='log', penalty = 'l2', alpha=0.01))])
	text_clf_svm = text_clf_svm.fit(sent_train, train_y)

	new_f_svm = text_clf_svm.predict_proba(sent_train)


	stemmer = SnowballStemmer('english', ignore_stopwords=True)
	class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer, self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	stemmed_count_vect = StemmedCountVectorizer(stop_words = 'english')
	text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
	                             ('mnb', MultinomialNB(fit_prior=False, alpha=0.05))])
	text_mnb_stemmed = text_mnb_stemmed.fit(sent_train, train_y)
	predicted_mnb_stemmed = text_mnb_stemmed.predict_proba(sent_train)

	text_mnb_stemmed_svm = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
	                             ('mnb', SGDClassifier(loss = 'log', penalty = 'l2', alpha=0.01))])
	text_mnb_stemmed_svm = text_mnb_stemmed_svm.fit(sent_train, train_y)
	predicted_mnb_stemmed_svm_train = text_mnb_stemmed_svm.predict_proba(sent_train)

	new_f = new_f_nb + new_f_svm + predicted_mnb_stemmed + predicted_mnb_stemmed_svm_train

	clf = RandomForestClassifier(max_depth = 4, random_state = 0)
	clf.fit(new_f, train_y)
	pred_rf = clf.predict_proba(new_f)
	new_f = map(sum, zip(new_f,pred_rf))

	clf2 = LogisticRegression(penalty = 'l2')
	clf2.fit(new_f + pred_rf, train_y)
	print '---------finish training----------------'	
	return text_clf, text_clf_svm, text_mnb_stemmed, text_mnb_stemmed_svm, clf, clf2

def testing(sent_test, nb_clf, svm_clf, stem_nb_clf, stem_svm_clf, rf_clf, lr_clf):
	test_f = nb_clf.predict_proba(sent_test) + svm_clf.predict_proba(sent_test) + stem_nb_clf.predict_proba(sent_test) + stem_svm_clf.predict_proba(sent_test) 
	test_f = map(sum, zip(test_f, rf_clf.predict_proba(test_f)))
	pred = lr_clf.predict(test_f)

	return pred 

def main():
	author = ['Jean Paul Marat', 'William Skeen', 'Thomas Hunt Morgan', 'Chas. H. Brown', 'James Tod','Russell A. Kelly', 'Augustus Le Plongeon', 'Kabir', 'Battiscombe G. Gunn', 'Jacob Kainen']
	pathes = ['corpus1.txt','corpus2.txt', 'corpus3.txt', 'corpus4.txt', 'corpus5.txt', 'corpus6.txt', 'corpus7.txt', 'corpus8.txt', 'corpus9.txt', 'corpus0.txt']
	times = [20, 5, 4, 10, 1, 8, 7, 10, 10, 7]
	sent_x, data_y, data_x = [], [], []

	#-----Read file
	for i, path in enumerate(pathes):
		for j in range(times[i]):
			sent = readTxt(path)
			sent_x.append(sent)
			data_y.append([author[i]] * len(sent))
	
	sent_x = list(np.concatenate(sent_x))
	data_y = list(np.concatenate(data_y))

	#-----Split data into training set and testing set
	X, Y = shuffle(sent_x, data_y, random_state = 0)
	sent_train, sent_test, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

	#-----Training, output the models
	nb_clf, svm_clf, stem_nb_clf, stem_svm_clf, rf_clf, lr_clf = training(sent_train, train_y)

	#-----Using the models to test on training and testing data
	pred_training = testing(sent_train, nb_clf, svm_clf, stem_nb_clf, stem_svm_clf, rf_clf, lr_clf)
	pred_test = testing(sent_test, nb_clf, svm_clf, stem_nb_clf, stem_svm_clf, rf_clf, lr_clf)

	print 'training', np.mean(pred_training == train_y)
	print 'TESTING ', np.mean(pred_test == test_y)

	from sys import argv
	print 'File with testing text'
	filename = argv[1]
	sent = readTxt(filename)	
	pred_result =  testing(sent, nb_clf, svm_clf, stem_nb_clf, stem_svm_clf, rf_clf, lr_clf)
	dic = {}
	for i in pred_result:
		if i not in dic:
			dic[i] = 1
		else:
			dic[i] += 1
	print sorted(dic, key=dic.get)[0]

 	
main()