from sklearn.naive_bayes import  GaussianNB
import numpy as np
dados = pd.read_csv('Iris_Floresta_Randomica.csv')


x = np.array([[1,2],[1,2],[1,2],[-2,0],[2,3],[-4,0],[-1,1],[1,2],[-2,2],[2,7],[-4,1],[0,0]])
y = np.array([1, 2, 6, 7, 2, 9, 3, 13, 8, 10, 4,7 ])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

model = GaussianNB()
model.fit(x_train, y_train)
teste = np.array([[2,7],[1,2],[-2,0],[-4,1]])
predicted = model.predict(teste)
print(predicted)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)
