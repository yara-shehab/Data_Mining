#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
class preprocessing :   
    def scale (self,method,X_train):
        if(method=="standardscaler"):
            stdsc = StandardScaler() 
            X_transformed = stdsc.fit_transform(X_train)
        if(method=="minmaxscaler"):
            mms = MinMaxScaler()
            X_transformed = mms.fit_transform(X_train)
        if(method=="maxscaler"):  
            max_abs_scaler = preprocessing.MaxAbsScaler()
            X_transformed = max_abs_scaler.fit_transform(X_train)
        if(method=="minscaler"):  
            max_abs_scaler = preprocessing.MaxAbsScaler()
            X_transformed = max_abs_scaler.fit_transform(X_train)   
        return X_transformed
    def encode (self,attribute,X_train):
        le = LabelEncoder()
        y= le.fit_transform(X_train[attribute].values)
        return y
    def encodey (self,y):
        le = LabelEncoder()
        y= le.fit_transform(y.values)
        return y
    def missing_data(self,X_train):
        X_train.dropna()   
        return X_train
class classifier :
    def __init__(self, method,X_train,X_test,y_train,y_test):
        if(method=="knn"):  
            knn = KNeighborsClassifier(n_neighbors=2)
            knn.fit(X_train, y_train)
            acc=knn.score(X_test,y_test)
            pred = knn.predict(X_test)
            print(acc)
            print(pred)
            
        if(method=="RandomForestClassifier"):  
            clf=RandomForestClassifier(n_estimators=80)##when number of estimators dec acc inc###
            clf.fit(X_train,y_train)
            acc= clf.score(X_test,y_test)
            pred=clf.predict(X_test)
            print(acc)
            print(pred)
        if(method=="DecisionTreeClassifier"):  
            clf= tree.DecisionTreeClassifier()
            clf.fit(X_train,y_train)
            acc = clf.score(X_test,y_test)
            pred=clf.predict(X_test)
            print(acc)
            print(pred)
        if(method=="BayesianClassifier"): 
            clf = GaussianNB()
            clf.fit(X_train,y_train)
            acc = clf.score(X_test,y_test)
            pred=clf.predict(X_test)
            print(acc)
            print(pred)
            
class regression :
    def __init__(self, method,X_train,X_test,y_train,y_test):
        if(method=="LinerRegression"):  
            lr = LinearRegression(normalize=True)
            lr.fit(X_train,y_train)
            acc = lr.score(X_test,y_test)
            predictions=lr.predict(X_test)
            print(acc)
            print(predictions)
        if(method=="PolynomialRegression"):        
            poly = PolynomialFeatures(2)
            X_poly = poly.fit_transform(X_train)
            poly_model = LinearRegression() 
            poly_model.fit(X_poly, y_train)
            pred = poly_model.predict(X_poly)
            print(acc)
        if(method=="DecisionTreeRegression"):  
            regressor = DecisionTreeRegressor(random_state = 0)  
            regressor.fit(X_train,y_train)
            pred = regressor.predict(X_test)
            acc = regressor.score(X_test,y_test)
            print(acc)
            print(pred)
        if(method=="KnnRegressor"):    
            neigh = KNeighborsRegressor(n_neighbors=5) ##when k increase the acc inc##
            neigh.fit(X_train, y_train)
            acc=neigh.score(X_test,y_test)
            pred = neigh.predict(X_test)  
            print(acc)
            print(pred)
class clustering :   
    def __init__(self,X_train,X_test,y_train,y_test):
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(X_train,y_train)
            pred=kmeans.predict(X_test)
            print(pred)
X = pd.read_csv('F:\College\Semester 7\Computer Vision\diamonds.csv')
y=X.pop('price')
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33)

####preprocessing######
preprocess = preprocessing()
def preproces(x):
    z=preprocess.encode('cut',x)
    x=x.assign(cut=z)
    z=preprocess.encode('color',x)
    x=x.assign(color=z)
    z=preprocess.encode('clarity',x)
    x=x.assign(clarity=z)
    x.isnull().sum()
    x=preprocess. missing_data(x)
    z=preprocess.scale("minmaxscaler",x)
    x=z
    return x
def preprocesy(y):
    y.isnull().sum()
    y=preprocess. missing_data(y)
    return y
X_train=preproces(X_train)
y_train=preprocesy(y_train)
X_test=preproces(X_test)
y_test=preprocesy(y_test)
#regres=regression("KnnRegressor",X_train,X_test,y_train,y_test)
#######classification####
X = pd.read_csv('F:\College\Semester 7\Computer Vision\data.csv')
y=X.pop('diagnosis')
X.pop('Unnamed: 32')
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33)
def preproces1(x): 
    x.isnull().sum()
    x=preprocess.missing_data(x)
    z=preprocess.scale("minmaxscaler",x)
    x=z
    return x
def preproces2(y):
    y.isnull().sum()
    y=preprocess. missing_data(y)
    y=preprocess.encodey (y)
    return y
X_train=preproces1(X_train)
y_train=preproces2(y_train)
X_test=preproces1(X_test)
y_test=preproces2(y_test)
#classification=classifier("BayesianClassifier",X_train,X_test,y_train,y_test)
#######cluster####
X = pd.read_csv('F:\College\Semester 7\Computer Vision\Iris.csv')
y=X.pop('Species')
#X.pop('Unnamed: 32')
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.33)
def preproces1(x): 
    x.isnull().sum()
    x=preprocess.missing_data(x)
    z=preprocess.scale("minmaxscaler",x)
    x=z
    return x
def preproces2(y):
    y.isnull().sum()
    y=preprocess. missing_data(y)
    y=preprocess.encodey (y)
    return y
X_train=preproces1(X_train)
y_train=preproces2(y_train)
X_test=preproces1(X_test)
y_test=preproces2(y_test)
cluster=clustering(X_train,X_test,y_train,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




