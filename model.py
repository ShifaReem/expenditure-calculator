import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import joblib

data=pd.read_csv('eve.csv')
print(data.head())

x=data.loc[:,['TIN','YEAR']]
y=data.loc[:,['agg']]
z=data.loc[:,['rev']]
a=data.loc[:,['cap' ]]
b=data.loc[:,['soc']]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=0)
x_train,x_test,z_train,z_test=train_test_split(x,z,test_size=.1,random_state=0)
x_train,x_test,a_train,a_test=train_test_split(x,a,test_size=.1,random_state=0)
x_train,x_test,b_train,b_test=train_test_split(x,b,test_size=.1,random_state=0)


model1=RandomForestRegressor()
model2=RandomForestRegressor()
model3=RandomForestRegressor()
model4=RandomForestRegressor()
model1.fit(x_train,y_train)
model2.fit(x_train,z_train)
model3.fit(x_train,a_train)
model4.fit(x_train,b_train)


#pickle file
model1.fit(x_train,y_train)
model1.score(x_test,y_test)
model2.fit(x_train,z_train)
model2.score(x_test,z_test)
model3.fit(x_train,a_train)
model3.score(x_test,a_test)
model4.fit(x_train,b_train)
model4.score(x_test,b_test)


pickle.dump(model1,open("model1.pkl","wb"))
pickle.dump(model2,open("model2.pkl","wb"))
pickle.dump(model3,open("model3.pkl","wb"))
pickle.dump(model4,open("model4.pkl","wb"))




