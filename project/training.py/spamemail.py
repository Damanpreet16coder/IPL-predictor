import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv('C:\VS Code\spam.csv')
df['Category']=df['Category'].map({'ham':0,'spam':1})
vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(df['Message'])
Y=df['Category']

classifier=DecisionTreeClassifier(criterion='entropy')
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
accuracy=accuracy_score(Y_test,Y_pred)
print(accuracy)
def prict_scam(message):
    message_vectorize=vectorizer.transform([message])
    pridiction=classifier.predict(message_vectorize)
    return 'Spam'if pridiction[0]==1 else 'Ham'
print(prict_scam("REMINDER FROM O2: To get 6.00 pounds free call credit and details of great offers pls 2 this text with your valid name, house no and postcode similar to this Offer expires soon!"))




