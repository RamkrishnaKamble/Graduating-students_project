import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle

data=pd.read_csv("student-mat.csv",sep= ";")
data = data.select_dtypes(exclude=['object'])
print(data.columns)
predict='G3'
df=data[['G1','G2','G3']]
df=shuffle(df)
x = np.array(df.drop([predict], 1))
y=np.array(df[predict])
best = 0
for _ in range(20):
    x_train, x_rem, y_train, y_rem = sklearn.model_selection.train_test_split(x, y, train_size = 0.6)
    x_valid, x_test, y_valid, y_test = sklearn.model_selection.train_test_split(x_rem, y_rem, test_size = 0.5)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_valid, y_valid)
    if acc > best:
        best = acc
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)


pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print("-------------------------")

predicted= linear.predict(x_test)
error=np.subtract(y_test,predicted)
ssd_sum=(np.sum(np.square(error)))
cost=(0.5/len(predicted))*(ssd_sum)
print(cost)
#results
#with ('studytime','failures', 'freetime', 'health', 'absences', 'G1','G2','G3') cost=0.679
#with("G1","G2") cost=0.54