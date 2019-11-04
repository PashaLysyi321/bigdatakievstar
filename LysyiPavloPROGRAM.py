import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('train_target.csv', sep=',')
df1 =  pd.read_csv('tabular_data.csv', sep=',',).fillna(0)
data  = pd.merge(left = df1, right = df, left_on= df1['ID'], right_on= df['ID'], copy = False)
data.drop(['ID_x', 'ID_y', 'PERIOD'], axis=1, inplace=True)
data = data.groupby(['key_0']).mean()


x_train = data.drop('TARGET', axis = 1)
y_train = data['TARGET']

logReg = LogisticRegression(solver = 'lbfgs',penalty='l2')
logReg.fit(x_train,y_train)
print(logReg.score(x_train,y_train))


testcsv = pd.read_csv('test_target.csv', sep=',')
x_test = pd.merge(left = df1, right = testcsv, left_on= df1['ID'], right_on= testcsv['ID'], copy = False)
x_test.drop(['ID_x', 'ID_y', 'PERIOD', 'SCORE'], axis=1, inplace=True)
x_test = x_test.groupby(['key_0']).mean()

y_test = logReg.predict_proba(x_test)[:,1]
idi = testcsv['ID'].values
d = {'ID':idi, 'SCORE': y_test}
answer = pd.DataFrame(data=d)
np.savetxt(r'C:/Users/lysyi/Desktop/bigdata/Тестове_завдання_2019/LysyiPavlo_test.txt',answer, fmt='%d,%f')