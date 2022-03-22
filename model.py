import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('wine.csv')
df['quality_cat'] = df.quality.map({'bad':0,'good':1})
df.drop('quality',axis=1,inplace=True)
X = df.drop('quality_cat',axis=1)
y = df.quality_cat
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

model = RandomForestClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test)
accuracy = model.score(X_test,y_test)
print("Accuracy of the model is : ",accuracy*100,"%")
pickle.dump(model,open('_model.pkl','wb'))

