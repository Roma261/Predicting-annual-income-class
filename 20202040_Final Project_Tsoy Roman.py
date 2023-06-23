#!/usr/bin/env python
# coding: utf-8

# # UNIST IE30301 Data Mining
# ## Final Project
# ### Tsoy Roman ID:20202040

# Hypothesis:In the United States, the number of men and women who have annual income >50K is the highest, when the relationship status of man - husband,and woman-wife.

# In[1]:


# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Loading data

# In[2]:


# Load Data to df
df = pd.read_csv('classification_project.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# Handling data

# In[5]:


df['country'].value_counts().plot(kind='bar') #mydat['age'].value_counts() counts the value and them plot it
plt.title('Barplot of Country')
plt.xlabel('country')
plt.ylabel('frequency')
plt.show()


# In[6]:


#Making dataset only with values for which, country is United-States
df = df[df['country'] == 'United-States']


# In[7]:


df.info()


# In[8]:


df['income'].value_counts()


# In[9]:


plt.bar(['>50K', '<=50K'], df['income'].value_counts(ascending=True))
plt.show()


# In[10]:


# Checking imbalance ratio
ratio = len(df[df['income'] == '<=50K']) / len(df[df['income'] == '>50K'])
print(f'Ratio between >50K : <=50K = 1:{ratio:.4f}')


# Checking a categorical feature using `value_counts()` method

# In[11]:


# Save all the categorical columns and store in a list called cat_feat
cat_feat = ['workclass', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex']


# In[12]:


for cat in cat_feat:
    print(f'[INFO] Feature {cat}')
    print(df[cat].value_counts())
    print('\n')


# EDA, analyzing data by drawing charts and tables, related to hypothesis

# In[13]:


df.groupby(['sex', 'relationship'])['income'].value_counts()

#it shows one inconsistent data


# In[14]:


pd.crosstab(df['sex'], df['income']=='>50K').plot(kind='bar', figsize=(15, 6))
plt.title('Income')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.legend(['<=50K', '>50K'])
plt.show()


# In[15]:


#Creating dataset with income >50K
df1=df[df['income']=='>50K']


# In[16]:


sns.countplot(data=df1[df1['sex']=='Female'],x="relationship")
plt.title('Barplot of Relationship', fontsize=18) #we can specify the title
plt.xlabel('relationship status', fontsize=16)  #we can name x-axis
plt.ylabel('frequency', fontsize=16) #we can name y-axis
plt.show()


# In[17]:


sns.countplot(data=df1[df1['sex']=='Male'],x="relationship")
plt.title('Barplot of Relationship', fontsize=18) #we can specify the title
plt.xlabel('relationship status', fontsize=16)  #we can name x-axis
plt.ylabel('frequency', fontsize=16) #we can name y-axis
plt.show()


# In[18]:


pd.crosstab(df['relationship'], df['income']).plot(kind='bar', figsize=(15, 6))
plt.title('Ratio between relationship status and income')
plt.xlabel('Relationship status')
plt.legend(['<=50K', '>50K'])
plt.show()


# ## Data preprocessing

# In[19]:


df.hist(bins=100, figsize=(20, 15))
plt.show()


# Studying the domain, I came to conlusion that capital gain and capital loss have to be included in annual come, so we do not drop these values.

# Capital gain and capital loss mean the same, but with different sign, so we substract them and assing it to capital gain variable to reduce the dimensions and avoid multicollinearity problem

# In[20]:


df['capital gain']=df['capital gain']-df['capital loss']


# In[21]:


df['capital gain'].value_counts()


# In[22]:


df.info()


# In[23]:


df = df.drop(['capital loss'], axis=1)


# In[24]:


df.info() #checking if the above column was dropped


# In[25]:


# As for all rows country is United States, we just drop this column
df = df.drop(['country'], axis=1)


# Also, one of the variables either education num or education should be removed to have less dimensions and avoid multicollinearity.

# In[26]:


pd.concat([df['education num'].value_counts() , df['education'].value_counts()], axis=0)


# In[27]:


df = df.drop(['education'], axis=1)


# In[28]:


df.info()


# ## Removing irrelevant rows

# In[29]:


df[df['sex']=='Male'][df['relationship']=='Wife']


# In[30]:


df=df.drop(4733)


# In[31]:


df.info()


# ## Missing data

# In[32]:


total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Number of missing values', '%'])
missing_data.head(11)


# In[33]:


df=df.replace('?', np.nan)


# Numeric variables

# In[34]:


df.loc[df["age"].isnull(), "age"]=round(df["age"].mean(),0)


# In[35]:


df.loc[df["hours per week"].isnull(), "hours per week"]=round(df["hours per week"].mean(),0)


# In[36]:


df.loc[df["capital gain"].isnull(), "capital gain"]=round(df["capital gain"].mean(),0)


# In[37]:


df.loc[df["fnlwgt"].isnull(), "fnlwgt"]=round(df["fnlwgt"].mean(),0)


# In[38]:


df.loc[df["education num"].isnull(), "education num"]=round(df["education num"].mean(),0)


# Categorical variables

# In[39]:


cat_feat = ['workclass','marital', 'occupation', 'relationship', 'race', 'sex']


# In[40]:


def miss_value(dataframe,colname):
     mode=dataframe[colname].mode()[0]
     dataframe[colname].fillna(mode,inplace=True)


# In[41]:


for columns in cat_feat:
    miss_value(df,columns)


# In[42]:


df


# In[43]:


total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Number of missing values', '%'])
missing_data.head(11)


# Checking for duplicates

# In[44]:


feat = ['workclass','marital', 'occupation', 'relationship', 'race', 'sex','age','hours per week','capital gain','education num','fnlwgt']


# In[45]:


df.drop_duplicates(subset =feat,
                     keep = 'first', inplace = True)
 


# In[46]:


df.info()


# In[47]:


df


# Train test split

# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


training_set, test_set = train_test_split(df, test_size=0.2, random_state=1, stratify=df[['income']])


# In[50]:


ratio = len(training_set[training_set['income'] == '<=50K']) / len(training_set[training_set['income'] == '>50K'])
print(f'Ratio of income in training set >50K : <=50K = 1:{ratio:.4f}')


# In[51]:


ratio = len(test_set[test_set['income'] == '<=50K']) / len(test_set[test_set['income'] == '>50K'])
print(f'Ratio of income in testing set >50K : <=50K = 1:{ratio:.4f}')


# In[52]:


df.info()


# In[53]:


# let us first separate numerical and categorical(nominal/ordinal) columns

nom_feat = ['workclass','marital', 'occupation', 'relationship', 'race', 'sex']
ord_feat = ['education num']
num_feat = ['age', 'hours per week', 'capital gain', 'fnlwgt']


# In[54]:


# copy data for preventing damage in raw training data
data = training_set.copy()
data1=test_set.copy()


# In[55]:


X_train = data.iloc[:,:-1]
y_train = data.iloc[:,-1]
X_test = data1.iloc[:,:-1]
y_test = data1.iloc[:,-1]


# One-hot encoding and ordinal encoding

# In[56]:


# import OneHotEncoder and OrdinalEncoder from sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# In[57]:


ohe = OneHotEncoder(sparse=False)
ohe.fit(X_train[nom_feat].values)
X_train_nom = ohe.transform(X_train[nom_feat].values)
X_test_nom=ohe.transform(X_test[nom_feat].values)
X_train_ord=X_train[ord_feat].values
X_test_ord=X_test[ord_feat].values


# In[58]:


X_train_ord


# In[59]:


X_test_nom


# Feature scaling

# In[60]:


# import scaler
from sklearn.preprocessing import StandardScaler


# In[61]:


scalar = StandardScaler()
scalar.fit(X_train[num_feat].values)
X_train_num = scalar.transform(X_train[num_feat].values)
X_test_num=scalar.transform(X_test[num_feat].values)


# In[62]:


X_train = np.concatenate([X_train_num, X_train_ord, X_train_nom], axis=1)
X_test= np.concatenate([X_test_num, X_test_ord, X_test_nom], axis=1)


# In[63]:


X_test.shape


# In[64]:


X_train.shape


# ## Training models

# Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
log_reg_cw = LogisticRegression(class_weight='balanced',max_iter=800)
log_reg_cw.fit(X_train, y_train)
accuracy_logreg=log_reg_cw.score(X_train, y_train)


# In[66]:


accuracy_logreg


# Decision Tree

# In[67]:


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(random_state=0,class_weight='balanced')
dectree.fit(X_train, y_train)
accuracy_dectree=dectree.score(X_train, y_train)


# In[68]:


accuracy_dectree


# In[69]:


from sklearn.tree import DecisionTreeClassifier
dectree = DecisionTreeClassifier(random_state=0,class_weight='balanced',max_depth=10)
dectree.fit(X_train, y_train)
accuracy_dectree=dectree.score(X_train, y_train)


# In[70]:


accuracy_dectree


# Random forest

# In[71]:


from sklearn.ensemble import RandomForestClassifier
randfor = RandomForestClassifier(n_estimators=100, random_state=0,class_weight='balanced',max_depth=10)
randfor.fit(X_train,y_train)
accuracy_randfor=randfor.score(X_train, y_train)


# In[72]:


accuracy_randfor


# Support Vector Machine

# In[73]:


from sklearn import svm
suppvm = svm.SVC(kernel='poly',class_weight='balanced')
suppvm.fit(X_train, y_train)
accuracy_suppvm=suppvm.score(X_train, y_train)


# In[74]:


accuracy_suppvm


# In[75]:


suppvm1 = svm.SVC(kernel='rbf',class_weight='balanced')
suppvm1.fit(X_train, y_train)
accuracy_suppvm1=suppvm1.score(X_train, y_train)


# In[76]:


accuracy_suppvm1


# k-NN

# In[77]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 15,weights='distance') 
knn.fit(X_train, y_train) 
accuracy_knn=knn.score(X_train,y_train)


# ## Evaluation

# 10-fold Cross-validation

# In[78]:


from sklearn.model_selection import cross_val_score
def validation(model, x, y):
    scores = cross_val_score(model, x, y, cv=10)
    return scores.mean()


# In[79]:


X = np.concatenate([X_train, X_test], axis=0)
y= np.concatenate([y_train, y_test], axis=0)


# In[80]:


val_logreg=validation(log_reg_cw,X,y)


# In[81]:


val_dectree=validation(dectree, X, y)


# In[82]:


val_randfor=validation(randfor, X, y)


# In[83]:


val_suppvm=validation(suppvm, X, y)


# In[84]:


val_suppvm1=validation(suppvm1, X, y)


# In[85]:


val_knn=validation(knn, X, y)


# In[86]:


evaluation = pd.DataFrame({'Model': ['Logistic regression', 'Decision tree', 'Random Forest',
              'SVM(poly)','SVM(rbf)', 'k-NN'],
                           'Accuracy Score on training set': [accuracy_logreg, 
              accuracy_dectree,accuracy_randfor, 
              accuracy_suppvm,accuracy_suppvm1, accuracy_knn],
                           'Accuracy using 10-fold cross validation': [val_logreg, 
              val_dectree,val_randfor, 
              val_suppvm,val_suppvm1, val_knn]})
evaluation = evaluation.set_index('Model')


# In[87]:


evaluation


# ROC-AUC curve

# In[88]:


from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, precision_recall_fscore_support,roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# In[89]:


#Logistic Regression
y_proba= log_reg_cw.predict_proba(X_test)[:, 1]
roc_auc_logreg=roc_auc_score(y_test, y_proba)
#Decision tree
y_proba1=dectree.predict_proba(X_test)[:, 1]
roc_auc_logreg1=roc_auc_score(y_test, y_proba1)
#Random Forest
y_proba2=randfor.predict_proba(X_test)[:, 1]
roc_auc_logreg2=roc_auc_score(y_test, y_proba2)
#k-NN
y_proba4=knn.predict_proba(X_test)[:, 1]
roc_auc_logreg4=roc_auc_score(y_test, y_proba4)


# In[90]:


from sklearn.metrics import auc, roc_curve
#Logistic Regression
fpr, tpr, threshold = roc_curve(y_test, y_proba,pos_label='>50K')
rocauc = auc(fpr, tpr)
#Decision Tree
fpr1,tpr1,threshold1= roc_curve(y_test,y_proba1,pos_label='>50K')
rocauc1 = auc(fpr1, tpr1)
#Random Forest
fpr2,tpr2,threshold2= roc_curve(y_test,y_proba2,pos_label='>50K')
rocauc2 = auc(fpr2, tpr2)
#k-NN
fpr4,tpr4,threshold4= roc_curve(y_test,y_proba4,pos_label='>50K')
rocauc4 = auc(fpr4, tpr4)


# In[91]:


plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (area={rocauc:.4f})')
plt.plot(fpr1, tpr1, label=f'Decision tree (area={rocauc1:.4f})')
plt.plot(fpr2, tpr2, label=f'Random Forest (area={rocauc2:.4f})')
plt.plot(fpr4, tpr4, label=f'k-NN(area={rocauc4:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()

