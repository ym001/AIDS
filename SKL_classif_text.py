#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  classifTextSKL.JeuxSmear4Smear.py
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
#  

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_20newsgroups

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.model_selection import cross_validate
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

import unicodedata

def lecture_du_jeu_de_20news():
		#categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
		categorie = ['sci.crypt', 'sci.electronics','sci.med']
		twenty_train = fetch_20newsgroups(subset='train',categories=categorie, shuffle=True, random_state=42)
		doc=twenty_train.data
		label=[]
		for i in range(len(twenty_train.target)):
			label.append([categorie[twenty_train.target[i]]])
		return doc,label,categorie
		
def clean_all(document):
	for i in range(len(document)):
		document[i] = re.sub(re.compile('<.*?>'), '', document[i])	#supprime balise html
		document[i] = document[i].lower()				#passe en minuscule
		document[i] = re.sub(re.compile('[^a-z]'), ' ', document[i])	#supprime les caracteres speciaux
		document[i] = re.sub("[ ]{2,}", " ", document[i])		#enleve les espaces successsifs
		document[i] = document[i].strip()				#enleve les espaces debut et fin
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

def list_label(label_jeu):
	label=[]
	for lab in label_jeu:
		for l in lab:
			if l not in label:
				label.append(l)
	return label	
		
def main(args):

	#importation des données
	doc_train,label_train,label=lecture_du_jeu_de_20news()

	doc_train	= np.asarray(doc_train)
	doc_train=pretraitement_document(doc_train)
	
	
	#numerisation des labels
	mlb = MultiLabelBinarizer(classes=(label))
	label_train_ft= mlb.fit_transform(label_train)
	
	#parametrage classifieur
	C=50
	k_fold=5
	scoring = ['precision_macro', 'recall_macro']

	#classifieur
	clf={}
	clf['MultinomialNB']			= OneVsRestClassifier(MultinomialNB(alpha=0.01))
	clf['LinearSVC']			= OneVsRestClassifier(LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='crammer_singer', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000))
	clf['SVC']				= OneVsRestClassifier(SVC())
	clf['SVC_Rbf']				= OneVsRestClassifier(SVC(C=C,gamma=0.1,cache_size=200,decision_function_shape='ovo',kernel='rbf'))
	clf['SVC_linear']			= OneVsRestClassifier(SVC(C=C,gamma=0.1,kernel='linear'))
	clf['SVC_poly']				= OneVsRestClassifier(SVC(C=C,gamma=0.1,kernel='poly',degree=4))
	clf['SVC_sigmoid']			= OneVsRestClassifier(SVC(C=C,gamma=0.1,kernel='sigmoid',degree=4))
	clf['DecisionTree']			= OneVsRestClassifier(DecisionTreeClassifier(max_depth=5))
	clf['RandomForest']			= OneVsRestClassifier(RandomForestClassifier(n_estimators=500))
	clf['AdaBoost']				= OneVsRestClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5, min_samples_leaf=1),learning_rate=1.,n_estimators=200,algorithm="SAMME"))
	clf['BernoulliNB']			= OneVsRestClassifier(BernoulliNB())
	clf['PassiveAggressiveClassifier']	= OneVsRestClassifier(PassiveAggressiveClassifier(loss='hinge',C=1.0,max_iter=50))
	clf['KNeighborsClassifier']		= OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10))
	clf['SGDClassifier']			= OneVsRestClassifier(SGDClassifier(alpha=.0001, max_iter=50,penalty="elasticnet"))
	clf['Perceptron']			= OneVsRestClassifier(Perceptron(max_iter=50))
		
	for classifier  in clf:
		classif= Pipeline([('vectorizer', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', clf[classifier])])
		classif.fit(doc_train, label_train_ft)

		#crossvalidation
		score=cross_validate(classif, doc_train, label_train_ft, cv=k_fold,scoring="accuracy",return_train_score=False,verbose=1)
		predicted = cross_val_predict(classif,doc_train, label_train_ft, cv=k_fold)
				
		accuracy_cv=score["test_score"].mean()
		standard_deviation=score["test_score"].std()
		
		print("{} Accuracy : {} ; ecart type : {}".format(classifier,accuracy_cv, standard_deviation))
		print("{}".format(score["test_score"] ))
	
	print('Fin du script...')
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
