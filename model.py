import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import joblib

data=pd.read_csv('e:\Parkinson-disease-prediction\datas\parkinsons.csv')
df=data.copy()
df.drop('name',axis='columns',inplace=True)


new_df=df[df['MDVP:Fhi(Hz)']<400]

q3=df['MDVP:Flo(Hz)'].quantile(0.75)
q1=df['MDVP:Flo(Hz)'].quantile(0.25)

iqr=q3-q1

upperrange=q3+1.5*iqr
lowerrange=q1-1.5*iqr

new_df=new_df[(new_df['MDVP:Flo(Hz)']<upperrange)& (new_df['MDVP:Flo(Hz)']>lowerrange)]

q3=df['MDVP:Jitter(Abs)'].quantile(0.75)
q1=df['MDVP:Jitter(Abs)'].quantile(0.25)

iqr=q3-q1

upperrange=q3+1.5*iqr
lowerrange=q1-1.5*iqr

new_df=new_df[(new_df['MDVP:Jitter(Abs)']<upperrange)& (new_df['MDVP:Jitter(Abs)']>lowerrange)]


q3=df['MDVP:Jitter(Abs)'].quantile(0.75)
q1=df['MDVP:Jitter(Abs)'].quantile(0.25)

iqr=q3-q1

upperrange=q3+1.5*iqr
lowerrange=q1-1.5*iqr

new_df=new_df[(new_df['MDVP:Jitter(Abs)']<upperrange)& (new_df['MDVP:Jitter(Abs)']>lowerrange)]


q3=df['MDVP:RAP'].quantile(0.75)
q1=df['MDVP:RAP'].quantile(0.25)

iqr=q3-q1

upperrange=q3+1.5*iqr
lowerrange=q1-1.5*iqr

new_df=new_df[(new_df['MDVP:RAP']<upperrange)& (new_df['MDVP:RAP']>lowerrange)]


q3=df['MDVP:PPQ'].quantile(0.75)
q1=df['MDVP:PPQ'].quantile(0.25)

iqr=q3-q1

upperrange=q3+1.5*iqr
lowerrange=q1-1.5*iqr

new_df=new_df[(new_df['MDVP:PPQ']<upperrange)& (new_df['MDVP:PPQ']>lowerrange)]


new_df=new_df[(stats.zscore(new_df['PPE']))<3]


target=new_df['status']
features=new_df.drop('status',axis='columns')

scaler=MinMaxScaler()
features_scaled=scaler.fit_transform(features)

target=target.reset_index(drop=True)
features=features.reset_index(drop=True)

xtrain,xtest,ytrain,ytest=train_test_split(features_scaled,target,test_size=0.2,random_state=1,stratify=target)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)

print(knn.score(xtest,ytest))


joblib.dump(knn,'output_models/parkinsons_model.sav')