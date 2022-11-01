import streamlit as st

from sklearn.naive_bayes import FlorestaRandomica
import pandas as st
dados = pd.read_csv('Iris_Floresta_Randomica.csv')




#from sklearn.naive_bayes import  GaussianNB
#import streamlit as st
#dados = pd.read_csv('Iris_Floresta_Randomica.csv')

classes = dados['Species'] 

#x = np.array([[1,2],[1,2],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,2],[-2,2],[2,7],[-4,1],[0,0]])
#y = np.array([1, 2, 6, 7, 2, 9, 3, 13, 8, 10, 4,7 ])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = NaveBayes()
model.fit(x_train, y_train)
teste = np.array([[2,7],[1,2],[-2,0],[-4,1]])
predicted = model.predict(teste)
print(predicted)
from sklearn import metrics
print("FlorestaRandomica Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)
