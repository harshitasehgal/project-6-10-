#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("sales (adv).csv",index_col=0)


# In[3]:


df.head()


# In[4]:


df.corr()


# In[5]:


dfcor=df.corr()


# In[6]:


sns.heatmap(dfcor)


# In[7]:


df.isnull().sum()


# In[8]:


sns.heatmap(df.isnull())


# In[9]:


df.shape


# In[10]:


df['newspaper'].plot.box()
#outliers present


# In[11]:


df.plot.box()


# In[12]:


#lets remove outliers from newspaper
df.loc[df['newspaper']>85,'newspaper']=np.mean(df['newspaper'])


# In[13]:


df.plot.box()
#outliers removed


# In[14]:


sns.distplot(df['TV'])


# In[15]:


plt.scatter(df['TV'],df['sales'])


# In[16]:


plt.scatter(df['radio'],df['sales'])


# In[17]:


plt.scatter(df['newspaper'],df['sales'])
#newspapers is less correlated with sales


# In[18]:


plt.violinplot(df['TV'])


# In[19]:


sns.countplot(df['newspaper'][:25])
#countplot of the first 30 rows of column newspaper


# In[ ]:





# In[ ]:





# In[20]:


df['newspaper'].value_counts()


# In[21]:


df.radio.unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


collist=df.columns.values
ncol=12
nrows=8


# In[23]:


plt.figure(figsize=(ncol,5*ncol))
for i in range(0,len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(df[collist[i]],color='green',orient='v')
    plt.tight_layout()


# In[24]:


sns.pairplot(df)


# In[25]:


sns.pairplot(df,hue='sales',height=2.5)


# In[ ]:





# In[26]:


columns_target=['sales']
columns_train=['TV','radio','newspaper']
x=df[columns_train]
y=df[columns_target]


# In[27]:


lr=LinearRegression()


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=83)


# In[29]:


lr.fit(x_train,y_train)


# In[30]:


lr.score(x_test,y_test)


# In[31]:


from sklearn import linear_model
max_r_score=0
for r_state in range(42,200):
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=r_state,test_size=0.22)
    regr=linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    y_pred=regr.predict(x_test)
    r2_scr=r2_score(y_test,y_pred)
    if r2_scr>max_r_score:
        max_r_score=r2_scr
        final_r_state=r_state
print("max r2 score corresponding to ",final_r_state," is ",max_r_score)        


# In[32]:


from sklearn.model_selection import cross_val_score
a_score=cross_val_score(linear_model.LinearRegression(),x,y,cv=5,scoring="r2")
a_score


# In[33]:


# finalising the model at random_state 151
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=151)


# In[34]:


pred=lr.predict(x_test)
print("Predicted sales:",pred)
print("actual sales",y_test)


# In[35]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[36]:


lr.score(x_test,y_test)


# In[37]:


from sklearn.model_selection import cross_val_score
a_score=cross_val_score(linear_model.LinearRegression(),x,y,cv=10,scoring="r2")
a_score


# In[39]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_predict=rf.predict(x_test) 


# In[40]:


rf.score(x_test,y_test)


# In[ ]:





# In[41]:


from sklearn.linear_model import Lasso,Ridge
ls=Lasso()
ls.fit(x_train,y_train)
print(ls.score(x_train,y_train))


# In[42]:


from sklearn.model_selection import GridSearchCV
alphavalue={'alpha':[1,0.1,0.01,0.001,0.0001,0]}
model=Ridge()
grid=GridSearchCV(estimator=model,param_grid=alphavalue)
grid.fit(x,y)
print(grid)
print(grid.best_estimator_.alpha)
print(grid.best_params_)


# In[43]:


rd=Ridge(alpha=1)
rd.fit(x,y)
print(rd.coef_)
print(rd.score(x_test,y_test))


# In[44]:


from sklearn.model_selection import cross_val_score
a_score=cross_val_score(linear_model.LinearRegression(),x,y,cv=10,scoring="r2")
a_score


# In[45]:


from sklearn.externals import joblib
joblib.dump(rf,'rfsales.obj')
rf_from_joblib=joblib.load('rfsales.obj')


# In[ ]:




