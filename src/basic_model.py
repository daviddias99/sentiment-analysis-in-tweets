import pandas as pd

import spacy
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from preprocess import preprocessor

custom_stopwords_1 = [":","\\", ".",",","-", "e", "f", "j", "p", "w"]
custom_stopwords_2 = ["abov", "ani", "becaus", "befor", "could", "doe", "dure", "might", "must", "onc"]
custom_stopwords_3 = ["&", "<", "user", ">", "repeated", "number", "repeat"]
custom_stopwords_4 = ["onli", "ourselv", "shall", "themselv", "veri", "whi", "woul", "would", "yourselv"]
custom_stopwords_5 = ['b', 'c', 'e', 'f', 'g', 'h', 'j', 'l', 'n', 'p', 'r', 'u', 'v', 'w']
custom_stopwords_6 = ["<user>","<repeated>", "<number>", "<repeat>"]

class Model(object):

    vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                # max_df=0.3,
                #  encoding='ascii',
                #  max_features=5000,
                min_df=2,
                ngram_range=(1, 2),
                tokenizer=lambda x: x,
                lowercase=False,
                stop_words = stopwords.words('english') + custom_stopwords_1  + custom_stopwords_3 + custom_stopwords_6 + custom_stopwords_2
                )

    def __init__(self, classifier):
        '''
            Classifier:
                - lr : Logistic Regression
                - svc : Support Vector Classification
                - naive: MultinomialNB aka Naive Bayes
                - perc: Perceptron
                - dtc: Decision Tree Classifier
                - rfc: Random Forest Classifier
        '''
        if(classifier == "lr"):
            clf = LogisticRegression(random_state=0, class_weight='balanced')#, solver='sag', C=0.8)
        elif(classifier == "svc"):
            clf = SVC(random_state=0, class_weight='balanced')
        elif(classifier == "naive"):
            clf = MultinomialNB()
        elif(classifier == "perc"):
            clf = Perceptron(random_state=0, class_weight='balanced')
        elif(classifier == "dtc"):
            clf = DecisionTreeClassifier(random_state=0, class_weight='balanced')
        elif(classifier == "rfc"):
            clf = RandomForestClassifier(random_state=0, class_weight='balanced')
        else:
            raise Exception('Classifier must be either lr, svc or naive, but was {}'.format(classifier))

        self.model = Pipeline([
            ('vect', self.vectorizer),
            ('clf', clf),
        ])

    def fit_multilabel(self,X_train, X_test, y_train, categories):
        y_hat = np.zeros((X_test.shape[0], len(categories)))

        for i in range(len(categories)):
            print('... Processing {}'.format(categories[i]))
            # train the model using X_dtm & y
            self.model.fit(X_train, y_train[:, i])
            # compute the testing accuracy
            y_hat[:, i] = self.model.predict(X_test)

        return y_hat

    @staticmethod
    def preprocess(name, dataset, inf = 'stem'):
        stemmer = SnowballStemmer(language='english')
        lemma = WordNetLemmatizer()

        desc = "PreProcessing dataset {}...".format(name)
        data = []
        i = 0
        for x in tqdm(dataset, desc=desc):
            i += 1
            result = []
            for t in preprocessor.pre_process_doc(x):

                if(inf == 'stem'):
                    result.append(stemmer.stem(t))
                elif(inf == 'lemma'):
                    result.append(lemma.lemmatize(t))
                else:
                    result.append(t)
            # prov = neg_sentence(' '.join(result))
            data.append(result)

        return np.array(data)

nlp = spacy.load("en_core_web_sm")

def neg_sentence(sentence):
    doc = nlp(sentence)
    pos_list = ['ADJ', 'ADV', 'AUX', 'VERB']
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
    negation_head_tokens = [token.head for token in negation_tokens]
    new_doc = []
    negated_tokens = []
    for token in negation_head_tokens:
        end = token.i
        start = token.head.i + 1
        left, right = doc[:start], doc[:end]
        negated_tokens = doc[start:end]
    for token in doc:
        if token in negated_tokens:
            # if token.pos_ in pos_list and token.text not in blacklist:

            # or you can leave out the blacklist and put it here directly
            if token.pos_ in pos_list and token.text not in [token.text for token in negation_tokens]:
                new_doc.append('not_'+token.text)
                continue
            else:
                pass
        new_doc.append(token.text)
    return new_doc