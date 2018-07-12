import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import collections
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, merge, Activation, Dropout, LSTM, Flatten
from keras.layers.embeddings import Embedding 
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import sequence
from sklearn import preprocessing, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def readTxt(path):
	f = open(path,'r').read()
	sentence = sent_tokenize(f)
	return [s.replace('\n', '') for s in sentence if len(s) != 0]

def dataset(words, num_words):
	count = [['UNK', 0]]
	count.extend(collections.Counter(words).most_common(num_words-1))
	dic = dict()
	for word, _ in count:
		dic[word] = len(dic)
	data = []
	unk_count = 0
	for word in words:
		if word in dic:
			idx = dic[word]
		else:
			count[0][1] += 1
			idx = 0
		data.append(idx)
	rev_dic = dict(zip(dic.values(), dic.keys()))
	return data, count, dic, rev_dic

def word_embedding(all_token, vocab_size):
	data, count, dic, rev_dic = dataset(all_token, vocab_size)

	win_size = 3
	vec_dim = 100
	epoch = 20

	valid_size = 16
	valid_win = 100
	valid_examples = np.random.choice(valid_win, valid_size, replace = False)

	sampling_table = sequence.make_sampling_table(vocab_size)
	couples, labels = skipgrams(data, vocab_size, window_size = win_size, sampling_table = sampling_table)
	word_target, word_context = zip(*couples)
	word_target = np.array(word_target, dtype="int32")
	word_context = np.array(word_context, dtype="int32")

	input_target = Input((1,))
	input_context = Input((1,))
	embedding = Embedding(vocab_size, vec_dim, input_length=1, name='embedding')

	target = embedding(input_target)
	target = Reshape((vec_dim, 1))(target)
	context = embedding(input_context)
	context = Reshape((vec_dim, 1))(context)

	similarity = merge([target, context], mode='cos', dot_axes=0)

	dot_product = merge([target, context], mode='dot', dot_axes=1)
	dot_product = Reshape((1,))(dot_product)
	# add the sigmoid output layer
	output = Dense(1, activation='sigmoid')(dot_product)

	model = Model(input=[input_target, input_context], output=output)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop')

	arr_1 = np.zeros((1,))
	arr_2 = np.zeros((1,))
	arr_3 = np.zeros((1,))
	for cnt in range(epoch):
	    idx = np.random.randint(0, len(labels)-1)
	    arr_1[0,] = word_target[idx]
	    arr_2[0,] = word_context[idx]
	    arr_3[0,] = labels[idx]
	    loss = model.train_on_batch([arr_1, arr_2], arr_3)

	print '-------finish embedding----------'
	embedding_vector = model.get_weights()[0]
	return dic, embedding_vector

def featurize(sent, dic, embedding_vector):
	rtn = []
	token = nltk.word_tokenize(sent.lower())
	for t in token:
		if t not in dic:
			t = 'UNK'
		if len(rtn) == 0:
			rtn = embedding_vector[dic[t]]
		else:
			rtn = map(sum, zip(rtn,embedding_vector[dic[t]]))
	return rtn

def train(train_x, test_x, train_y, test_y, embedding_vector):
	train_x = np.array(train_x)
	test_x = np.array(test_x)
	train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
	test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
	model = Sequential()
	model.add(LSTM(256, input_shape=(1,100), return_sequences=True))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	model.fit(train_x, train_y, batch_size=16, epochs=200)
	score = model.evaluate(test_x, test_y, batch_size=16)
	print 'score', score

def main():
	author = ['Jean Paul Marat', 'William Skeen', 'Thomas Hunt Morgan', 'Chas. H. Brown', 'James Tod','Russell A. Kelly', 'Augustus Le Plongeon', 'Kabir', 'Battiscombe G. Gunn', 'Jacob Kainen']
	pathes = ['corpus1.txt','corpus2.txt', 'corpus3.txt', 'corpus4.txt', 'corpus5.txt', 'corpus6.txt', 'corpus7.txt', 'corpus8.txt', 'corpus9.txt', 'corpus0.txt']
	sent_x, data_y, data_x = [], [], []
	for i, path in enumerate(pathes):
		sent = readTxt(path)
		sent_x.append(sent)
		data_y.append([author[i]] * len(sent))

	sent_x = list(np.concatenate(sent_x))
	data_y = list(np.concatenate(data_y))
	all_sent = ' '.join(sent_x)
	print len(all_sent)
	all_token = nltk.word_tokenize(all_sent.lower())
	vocab_size = 10000
	dic, embedding_vector = word_embedding(all_token, vocab_size)
	print len(sent_x)
	
	data_x = []
	for sent in sent_x:
		tmp = featurize(sent, dic, embedding_vector)
		if len(tmp) == 0:
			print sent
		data_x.append(tmp)


	le = preprocessing.LabelEncoder()
	le.fit(data_y)
	data_y = le.transform(data_y)
	
	X, Y = shuffle(data_x, data_y, random_state = 0)
	train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)
	#train(train_x, test_x, train_y, test_y, embedding_vector)


	clf = RandomForestClassifier(max_depth = 8, random_state = 0)
	clf.fit(train_x, train_y)
	pred = clf.predict(test_x)
	print 'RF', accuracy_score(test_y, pred)

	clf2 = LogisticRegression(penalty = 'l1')
	clf2.fit(train_x, train_y)	
	pred = clf2.predict(test_x)
	print 'LR', accuracy_score(test_y, pred)

	clf3 = neighbors.KNeighborsClassifier(20, weights='distance')
	clf3.fit(train_x, train_y)	
	pred = clf3.predict(test_x)
	print 'KNN', accuracy_score(test_y, pred)


	






main()