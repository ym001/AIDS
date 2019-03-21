#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  deep_classif_text_multiclasse.py
#  
#  Copyright 2017 yves <yves.mercadier@lirmm.fr>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  

from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Activation, Lambda,SpatialDropout1D,Reshape,Concatenate, concatenate,BatchNormalization

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.convolutional import Convolution1D
from keras.layers import Conv1D,Conv2D, GlobalMaxPooling1D, MaxPooling1D,MaxPooling2D,AveragePooling1D,ZeroPadding1D
from keras.layers import Bidirectional,GlobalMaxPool1D
from keras.regularizers import l2

from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras import optimizers

import numpy as np
from keras.datasets import imdb
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import KFold

def lecture_du_jeu_de_imdbkeras():
		max_words=20000
		x_train_imdb=[]
		y_train_imdb=[]
		(x_imdb, y_imdb), (x_test_imdb, y_test_imdb) = imdb.load_data(num_words=max_words,maxlen=300,seed=113)
		
		#reconstruction
		wordDict = {y:x for x,y in imdb.get_word_index().items()}
		for doc in x_imdb:
			sequence=""
			for index in doc:
				sequence+=" "+wordDict.get(index)
			x_train_imdb.append(sequence)
		for i in y_imdb:
			y_train_imdb.append(str(i))
					
		return x_train_imdb,y_train_imdb
		
def clean_all(document):
	for i in range(len(document)):
		document[i] = re.sub(re.compile('<.*?>'), '', document[i])	#supprime balise html
		document[i] = document[i].lower()							#passe en minuscule
		document[i] = re.sub(re.compile('[^a-z]'), ' ', document[i])#supprime les caracteres speciaux
		document[i] = re.sub("[ ]{2,}", " ", document[i])			#enleve les espaces successsifs
		document[i] = document[i].strip()							#enleve les espaces debut et fin
	return document

def stopWords(document):
	stop_words = set(stopwords.words('english'))

	for i in range(len(document)):
		word_tokens=document[i].split(" ")
		filtered_phrase = []
		for w in word_tokens:
			if w not in stop_words:
				filtered_phrase.append(w)
		
		document[i]=" ".join(filtered_phrase)
	return document
		
def lemmatization(document):
	
	lemmatizer = WordNetLemmatizer()
	for i in range(len(document)):
		word_tokens=document[i].split(" ")
		lemm_phrase = []
		for w in word_tokens:
			lemm_phrase.append(lemmatizer.lemmatize(w))
		document[i]=" ".join(lemm_phrase)
	return document	
	
def pretraitement_document(document):
	print("pretraitement")
	document=clean_all(document)	#supprimetout les caracteres spéciaux.
	document=stopWords(document)	#stopWord
	document=lemmatization(document)
	print("fin pretraitement")
	return document
	
def pretraitement_deep(document,max_words):
	print("Pretraitement deep.")
	count_mot=0
	nb_mot_max=0
	word_index=[]

	encoded_docs= [one_hot(d, max_words) for d in document]

	##############
	#calcul de la taille des sequences
	count_sequence={}
	max_longueur=0
	for sequence in encoded_docs:
		longueur=len(sequence)
		if longueur in count_sequence:
			count_sequence[longueur]+=1
			if max_longueur<longueur:max_longueur=longueur
		else:count_sequence[longueur]=1
	somme_sequence=0
	for key in sorted(count_sequence.keys()):
		if somme_sequence<len(document)*0.9:
			somme_sequence+=count_sequence[key]
			longueur_sequence=key
	padded_docs = pad_sequences(encoded_docs, maxlen=longueur_sequence, padding='post')
	return padded_docs

def list_label(label_jeu):
	label=[]
	for lab in label_jeu:
		for l in lab:
			if l not in label:
				label.append(l)
	return label	

def numerisation_label(label_a_numeriser,label_list):

	num_classes=len(label_list)
	n=len(label_a_numeriser)
	categorical = np.zeros((n, num_classes))
	count=0
	for lab in label_a_numeriser:
		for classe in lab:
			index=label_list.index(classe)
			categorical [count,index]=1
		count=count+1
	return categorical
class Reseau:  
	
	def __init__(self,max_words): # constructeur
		self.max_words=max_words
		self.embedding_dims=300
		self.lstm_output=300
		self.batch_sizes=64
	def construct_modele(self,modele):
		self.modele=modele
		if self.modele=="lstm1":self.construct_lstm1()
		if self.modele=="lstm2":self.construct_lstm2()
		if self.modele=="rnn":self.construct_rnn()
		if self.modele=="lstmb":self.construct_lstmb()
		if self.modele=="lstmbs":self.construct_lstmbs()
		if self.modele=="cnn1":self.construct_cnn1()
		if self.modele=="cnn2":self.construct_cnn2()
		if self.modele=="mlp":self.construct_mlp()
		if self.modele=="aclstm":self.construct_aclstm()
		if self.modele=="mgnccnn":self.construct_mgnccnn()
		if self.modele=="MultiplicativeLSTM":self.construct_MultiplicativeLSTM()
		if self.modele=="clstm":self.construct_clstm()
		
	def construct_lstm1(self):
		# define the model
		print('Build model Lstm1')
		inputs = Input(shape=(self.input_dim,))
		if self.embedding=="keras":
			l1=Embedding(self.max_words,self.embedding_dims,input_length=self.input_dim)(inputs)
			l1=SpatialDropout1D(Dropout(self.dropout))(l1)
		else:l1=inputs
		outputs=LSTM(
								self.lstm_output,
								return_sequences=False,
								consume_less=self.consume_less,
								dropout=self.dropout,
								recurrent_dropout=self.dropout,
								activation=self.activation_hidden_size,
								inner_activation=self.inner_activation,
								W_regularizer=None,
								U_regularizer=None,
								b_regularizer=None,
								init=self.init,
								inner_init=self.inner_init,
								forget_bias_init=self.forget_bias_init
								)(l1) 
		outputs=Dense(self.input_dim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)

	def construct_lstm2(self):
		# define the model
		print('Build model Lstm2')
		inputs = Input(shape=(self.input_dim,))
		if self.embedding=="keras":
			l1=Embedding(self.max_words,self.embedding_dims,input_length=self.p.input_dim)(inputs)
			l1=SpatialDropout1D(Dropout(self.dropout))(l1)
		else:l1=inputs

		#outputs=LSTM(self.input_dim*32, dropout=self.dropout, recurrent_dropout=self.dropout)(l1) 
		outputs=LSTM(self.lstm_output, dropout=self.dropout, recurrent_dropout=self.dropout)(l1) 
		outputs=Dense(self.input_dim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)


	def construct_rnn(self):
		# define the model
		print('Build model Lstm2')
		inputs = Input(shape=(self.input_dim,))
		if self.embedding=="keras":
			l1=Embedding(self.max_words,self.embedding_dims,input_length=self.p.input_dim)(inputs)
			l1=SpatialDropout1D(Dropout(self.dropout))(l1)
		else:l1=inputs
		outputs=SimpleRNN(self.input_dim*20, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)(l1)

		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)

	def construct_lstmb(self):
		# define the model
		print('Build model Lstm bidirectionnel:')
		
		inputs = Input(shape=(self.input_dim,))
		embedded_layer=self.embedding_layer(inputs)

		outputs=Bidirectional(LSTM(self.lstm_output, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(embedded_layer) 
		outputs=Bidirectional(LSTM(self.lstm_output, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(embedded_layer) 
		outputs = GlobalMaxPool1D()(outputs)
		
		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)
		
	def construct_lstmbs(self):
		# define the model
		print('Build model Lstm bidirectionnel suivi:')
		inputs = Input(shape=(self.input_dim,))
		if self.embedding=="keras":
			l1=Embedding(self.max_words,self.embedding_dims,input_length=self.input_dim)(inputs)
			l1=SpatialDropout1D(Dropout(self.dropout))(l1)
		else:l1=inputs
		outputs=Bidirectional(LSTM(self.lstm_output, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(l1) 
		outputs=Bidirectional(LSTM(self.lstm_output, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(l1) 
		#outputs=LSTM(self.lstm_output, return_sequences=True, dropout=0.5, recurrent_dropout=0.1)(l1) 
		outputs = GlobalMaxPool1D()(outputs)
		
		outputs=Dense(self.lstm_output/2, activation=self.activation_hidden_size,kernel_initializer='normal')(outputs)
		outputs=Dense(self.lstm_output/4, activation=self.activation_hidden_size,kernel_initializer='normal')(outputs)
		outputs=Dense(self.lstm_output/8, activation=self.activation_hidden_size,kernel_initializer='normal')(outputs)

		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)
	
	def construct_cnn1(self):
		print('Build model CNN1')
		#http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/11/27/Understanding-Convolutions-In-Text/
		inputs = Input(shape=(self.input_dim, self.max_words), name='input', dtype='float32')

		outputs = Convolution1D(nb_filter=self.filters, filter_length=1,border_mode='valid', activation='relu')(inputs)
		outputs = MaxPooling1D(pool_length=3)(outputs)
		outputs = Flatten()(outputs)

		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)
	def construct_cnn2(self):
		#https://github.com/Jverma/cnn-text-classification-keras
		#paper
		print('Build model CNN2')
		inputs = Input(shape=(self.input_dim,), dtype='int32')
		embedded_layer=self.embedding_layer(inputs)

		# add first conv filter
		embedded_layer = Reshape((self.input_dim, self.embedding_dims, 1))(embedded_layer)
		x = Conv2D(200, (5, self.embedding_dims), activation='relu')(embedded_layer)
		x = MaxPooling2D((self.input_dim - 5 + 1, 1))(x)


		# add second conv filter.
		y = Conv2D(200, (4, self.embedding_dims), activation='relu')(embedded_layer)
		y = MaxPooling2D((self.input_dim - 4 + 1, 1))(y)


		# add third conv filter.
		z = Conv2D(200, (3, self.embedding_dims), activation='relu')(embedded_layer)
		z = MaxPooling2D((self.input_dim - 3 + 1, 1))(z)


		# concate the conv layers
		alpha = concatenate([x,y,z])

		# flatted the pooled features.
		alpha = Flatten()(alpha)

		# dropout
		outputs = Dropout(0.5)(alpha)

		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)
	
	def construct_aclstm(self):
		#https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/deep_models/paper_16_ac_blstm/models.ipynb
		#paper
		print('Build model CNN2')
		inputs = Input(shape=(self.input_dim,), dtype='int32')
		embedded_layer=self.embedding_layer(inputs)

		convs = []
		ngram_filters = [30, 40, 50, 60]
		n_filters = 64

		for n_gram in ngram_filters:
			l_conv1 = Conv1D(filters = n_filters, kernel_size = 1, strides = 1,padding="same")(embedded_layer)
			#     l_batch1 = BatchNormalization()(l_conv1)
			l_relu1 = Activation("relu")(l_conv1)
			l_conv2 = Conv1D(filters = n_filters, kernel_size = n_gram, strides = 1,padding="same")(l_relu1)
			#     l_batch2 = BatchNormalization()(l_conv2)
			l_relu2 = Activation("relu")(l_conv2)
			convs.append(l_relu2)

		l_concat = Concatenate(axis=2)(convs)
		
		l_blstm = Bidirectional(LSTM(32, activation="relu", return_sequences=True))(l_concat)
		l_dropout = Dropout(0.5)(l_blstm)
		outputs = Flatten()(l_dropout)
		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)
		
	def construct_mgnccnn(self):
		#https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/deep_models/paper_07_mgnccnn/models.ipynb
		#paper
		print('Build model mgnccnn')
		WORD_EMB_SIZE1 = 300
		WORD_EMB_SIZE2 = 200
		WORD_EMB_SIZE3 = 100
		inputs = Input(shape=(self.input_dim,), dtype='int32')
		text_embedding1 =Embedding(self.max_words,WORD_EMB_SIZE1,input_length=self.input_dim)(inputs)
		text_embedding2 =Embedding(self.max_words,WORD_EMB_SIZE2,input_length=self.input_dim)(inputs)
		text_embedding3 =Embedding(self.max_words,WORD_EMB_SIZE3,input_length=self.input_dim)(inputs)

		k_top = 4
		filter_sizes = [3,5]

		conv_pools = []
		for text_embedding in [text_embedding1, text_embedding2, text_embedding3]:
			for filter_size in filter_sizes:
				l_zero = ZeroPadding1D((filter_size-1,filter_size-1))(text_embedding)
				l_conv = Conv1D(filters=16, kernel_size=filter_size, padding='same', activation='tanh')(l_zero)
				l_pool = GlobalMaxPool1D()(l_conv)
				conv_pools.append(l_pool)
  
		l_merge = Concatenate(axis=1)(conv_pools)
		outputs = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(l_merge)
		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)

	def construct_MultiplicativeLSTM(self):
		#https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/deep_models/paper_10_mul_lstm/models.ipynb
		#paper
		print('Build model MultiplicativeLSTM')
		inputs = Input(shape=(self.input_dim,), dtype='int32')
		embedded_layer=self.embedding_layer(inputs)
		from utils import MultiplicativeLSTM
		outputs=MultiplicativeLSTM(32)(embedded_layer)
		
		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)

		self.model=Model(inputs=inputs,outputs=outputs)
		
	def construct_clstm(self):
		#https://github.com/bicepjai/Deep-Survey-Text-Classification/blob/master/deep_models/paper_14_clstm/models.ipynb
		#paper
		print('Build model clstm')
		inputs = Input(shape=(self.input_dim,), dtype='int32')
		embedded_layer=self.embedding_layer(inputs)
		convs = []
		filter_sizes = [10, 20, 30, 40, 50]
		for filter_size in filter_sizes:
			l_conv = Conv1D(filters=64, kernel_size=filter_size, padding='valid', activation='relu')(embedded_layer)
			convs.append(l_conv)

		cnn_feature_maps = Concatenate(axis=1)(convs)
		sentence_encoder = LSTM(64,return_sequences=False)(cnn_feature_maps)
		outputs =Dense(128, activation="relu")(sentence_encoder)
		outputs=Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)

		self.model=Model(inputs=inputs,outputs=outputs)

	def construct_mlp(self):
		# define the model
		print('Build model MLP')
		inputs = Input(shape=(self.input_dim,))
		if self.embedding=="keras":
			l1=Embedding(self.max_words,self.embedding_dims,input_length=self.input_dim)(inputs)
		else:
			l1=inputs

		outputs=Dense(self.hidden_size, activation=self.activation_hidden_size,kernel_initializer='normal')(l1)
		outputs = Dropout(0.5)(outputs)

		for i in range(2):
			outputs=Dense(self.hidden_size, activation=self.activation_hidden_size,kernel_initializer='normal')(outputs)
			outputs = Dropout(0.5)(outputs)
		
		outputs=self.couche_de_sortie(outputs)
		self.model=Model(inputs=inputs,outputs=outputs)
		
	def couche_de_sortie(self,outputs):
		return Dense(self.outputdim, activation=self.activation_sortie,kernel_initializer='normal')(outputs)

		
	def compile(self):
		self.model.compile(loss=self.loss_fonction, optimizer=self.optimizer(),metrics=['accuracy','categorical_accuracy'])
	
	def optimizer(self):
		self.optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
		return self.optimizer
		
	def predict(self,x):
		return self.model.predict(x)
		
	def fit(self):
		self.model.fit(self.x, self.y,batch_size=self.batch_sizes,epochs=self.epoch,verbose=1,shuffle=True ,validation_split=0.0, validation_data=None)
					
	def setX(self,x):
		self.x=x
	def setY(self,y):
		self.y=y
		
	def setDropout(self,d):
		self.dropout=d
	def setEpoch(self,e):
		self.epoch=e
	def setModele(self,m):
		self.modele=m
	def setLossFonction(self,loss):
		self.loss_fonction=loss

	def setInputDim(self,input_dim):
		self.input_dim=input_dim
	def setOutputDim(self,outputdim):
		self.outputdim=outputdim
	def setHiddenSize(self,hidden_size):
		self.hidden_size=hidden_size
	def setActivationSortie(self,activation_sortie):
		self.activation_sortie=activation_sortie
		
	def setActivationHiddenSize(self,activation_hidden_size):
		self.activation_hidden_size=activation_hidden_size

	def setEmbedding(self,embedding):
		self.embedding=embedding

	def setxtrain(self,xtrain):
		self.xtrain=xtrain
		
	def get_model(self):
		return self.model
		
	def shuffle_weights(self, weights=None):
		weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
		self.model.set_weights(weights)
		
	def embedding_layer(self,l):
		if self.embedding=="glove":
			embeddings_index=indexingWordVectors()
			embedding_matrix=prepareEmbeddingMatrix(embeddings_index,self.p.max_words,self.word_index)
			embedding = Embedding(self.max_words,self.embedding_dims,weights=[embedding_matrix],input_length=self.input_dim,trainable=True)(l)
			print("Embedding "+self.embedding)
		if self.embedding=="fasttext":
			from embedding import constructionModeleFasttext
			embedding_matrix=constructionModeleFasttext(self.xtrain,self.jeux,self.max_words,self.word_index,self.embedding_dims)
			embedding = Embedding(self.max_words,self.embedding_dims,weights=[embedding_matrix],input_length=self.input_dim,trainable=True)(l)
			print("Embedding "+self.embedding)

		if self.embedding=="keras":
			embedding=Embedding(self.max_words,self.embedding_dims,input_length=self.input_dim)(l)
			print("Embedding "+self.embedding)
			
		embedding=SpatialDropout1D(Dropout(self.dropout))(embedding)

		return embedding

def main(args):
		
	#importation des données
	doc_train,label_train=lecture_du_jeu_de_imdbkeras()
	label_list=list_label(label_train)
	doc_train	= np.asarray(doc_train)
	doc_train=pretraitement_document(doc_train)		
	max_words=10000;
	doc_train=pretraitement_deep(doc_train,max_words)		
	label_train=numerisation_label(label_train,label_list)

	#classif multiclasse standart
	reseau_multiclasse=Reseau(max_words)
	reseau_multiclasse.setInputDim(doc_train.shape[1])
	reseau_multiclasse.setOutputDim(len(label_train[0]))
	reseau_multiclasse.setActivationSortie('softmax')
	reseau_multiclasse.setEmbedding("keras")
	reseau_multiclasse.setDropout(0.4)
	reseau_multiclasse.setEpoch(25)
	reseau_multiclasse.setLossFonction('categorical_crossentropy')
	#name_model="aclstm"
	#name_model="cnn2"
	#name_model="mgnccnn"
	#name_model="MultiplicativeLSTM"
	#name_model="clstm"
	name_model="lstmb"
	#name_model="lstmbs"
	#name_model="lstm2"
	reseau_multiclasse.construct_modele(name_model)
	reseau_multiclasse.compile()
	init_weight =reseau_multiclasse.get_model().get_weights()

	#init cross validation
	xfold = np.zeros(len(doc_train))
	k_fold=5
	skf = KFold(n_splits=k_fold)
	skf.get_n_splits(xfold)
	acc_cross_validation=[]
	i=0
	for train_indices, val_indices in skf.split(xfold):
		print("passe : {}".format(i))
		i+=1
		reseau_multiclasse.shuffle_weights(init_weight)
		x_train, x_validation = doc_train[train_indices], doc_train[val_indices]
		y_train, y_validation = label_train[train_indices],label_train[val_indices]
	
		reseau_multiclasse.setX(x_train)
		reseau_multiclasse.setY(y_train)
	
		reseau_multiclasse.fit()

		y_prediction=reseau_multiclasse.predict(x_validation)

		acc_cross_validation.append(acc_multiclasse(y_validation,y_prediction))
		print("{} - {} - acc multiclasse : {} - {}".format(name_model,xtrain.shape[0],np.mean(acc_cross_validation),acc_cross_validation))

	print('Fin du script...')
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))


