import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('dataset4.csv')
data.head()


data.shape


data['language'].value_counts()


data['language'][2]


data['Text'][3]


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

ps = PorterStemmer()
corpus=[]

for i in range(len(data['Text'])):
    
    rev = re.sub("^[a-zA-Z]",' ', data['Text'][i]) 
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if set(stopwords.words())]
    rev = ' '.join(rev)
    corpus.append(rev)
    
    print(f"{i}")



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)
X = cv.fit_transform(corpus).toarray()


X.shape



from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y = label.fit_transform(data['language'])


y


len(y)

label.classes_


data1 = pd.DataFrame(np.c_[corpus,y],columns=['Sentence','Language'])

data1


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.naive_bayes import MultinomialNB


classifier = MultinomialNB().fit(X_train,y_train)


pred = classifier.predict(X_test)


pred

y_test


from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))


plt.figure(figsize=(20,20))
sns.heatmap(confusion_matrix(y_test,pred),annot=True,cmap=plt.cm.Accent)


fnl = pd.DataFrame(np.c_[y_test,pred],columns=['Actual','Predicted'])
fnl


import joblib

joblib.dump(classifier , 'language_identification.sav')


model = joblib.load('language_identification.sav')


def test_model(test_sentence):
    languages = {
    'English' : 0,
    'Russian' : 1,
    'Thai' : 2,
    'Turkish' : 3
    }
    
    
    
    
    rev = re.sub('^[a-zA-Z]',' ',test_sentence)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if word not in set(stopwords.words())]
    rev = ' '.join(rev)
    
    rev = cv.transform([rev]).toarray()
    
    output = model.predict(rev)[0]
    
    keys = list(languages)
    values = list(languages.values())
    position = values.index(output)
    
    output = keys[position]
    
    print(output)


test_model('พวกเราเป็นเด็กดี')

test_model('İyi ki doğdun Raman') 

test_model('это модель прогнозирования дома')

test_model('this is good product')



