import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os, ipaddress
from sklearn import preprocessing


df = pd.read_csv(os.path.join(os.path.dirname(__file__),'dataset.csv'))
df.dropna(inplace=True)

numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print("The number of numerical features is",len(numerical_features),"and they are : \n",numerical_features)

categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
print("The number of categorical features is",len(categorical_features),"and they are : \n",categorical_features)

#discrete numerical features 
discrete_feature = [feature for feature in numerical_features if df[feature].nunique()<=15 and feature != 'label']
print("The number of discrete features is",len(discrete_feature),"and they are : \n",discrete_feature)

continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature + ['label']]
print("The number of continuous_feature features is",len(continuous_feature),"and they are : \n",continuous_feature)

import warnings
warnings.filterwarnings("ignore")
for feature in continuous_feature:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['pktcount']=np.log(data['pktcount'])
        plt.figure(figsize=(20,20))
        f = sns.relplot(data=data, x=data[feature],y=data['pktcount'],hue="Protocol",style="Protocol",
                    col="label",kind="scatter").set(title="Диаграмма рассеивания количества пакетов")
        f.set_ylabels("Количество пакетов", clear_inner=False)
        f.set_xlabels("Время, номер дня", clear_inner=False)

bridge_types = ('UDP','TCP','ICMP')
le = preprocessing.LabelEncoder()

df['Protocol'] = le.fit_transform(df['Protocol'])
df['src'] = df['src'].apply(lambda x: int(ipaddress.IPv4Address(x)))
df['dst'] = df['dst'].apply(lambda x: int(ipaddress.IPv4Address(x)))



malign = df[df['label'] == 1]
benign = df[df['label'] == 0]

# Let's plot the Label class against the Frequency
labels = ['Норма','DDoS']
classes = pd.value_counts(df['label'], sort = True) / df['label'].count() *100
classes.plot(kind = 'bar')
plt.title("Распределение классов")
plt.xticks(range(2), labels)
plt.xlabel("Класс")
plt.ylabel("Частота %")

df['label'] = df['label'].apply(lambda x:  'Норма' if x == 0 else 'DDoS')
pp = sns.pairplot(df,hue="label",vars=['dur','flows','byteperflow'])
#pp = sns.pairplot(df,hue="label",vars=['pktcount','flows','bytecount'])
df['label'] = df['label'].apply(lambda x:  0 if x == 'Норма' else 1)

print(df.apply(lambda col: col.unique()))



# def countplot_distribution(col):
#     sns.set_theme(style="darkgrid")
#     sns.countplot(y=col, data=df).set(title = 'Distribution of ' + col)

# def histplot_distribution(col):
#     sns.set_theme(style="darkgrid")
#     sns.histplot(data=df,x=col, kde=True,color="red").set(title = 'Distribution of ' + col)

# ## Lets analyse the categorical values by creating histograms to understand the distribution
# f = plt.figure(figsize=(8,20))
# for i in range(len(categorical_features)):
#     f.add_subplot(len(categorical_features), 1, i+1)
#     countplot_distribution(categorical_features[i])
# plt.show()

# def get_percentage_malign_protocols():
#     arr = [x for x, y in zip(df['Protocol'], df['label']) if y == 1]
#     perc_arr = []
#     for i in [0,1,2]:
#         perc_arr.append(arr.count(i)/len(arr) *100)
#     return perc_arr
# #distribution of protocols
# fig1, ax1 = plt.subplots(figsize=[7,7])
# ax1.pie(get_percentage_malign_protocols(), explode=(0.1, 0, 0), autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')
# ax1.legend(['UDP', 'TCP', 'ICMP'],loc="best")
# plt.title('Процентное распределение протоколов в DDoS трафике',fontsize = 14)
# plt.show()

#correlation matrix
correlation_matrix = df.corr()
fig = plt.figure(figsize=(17,17))
mask = np.zeros_like(correlation_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
sns.set_theme(style="darkgrid")
ax = sns.heatmap(correlation_matrix,square = True,annot=True,center=0,vmin=-1,linewidths = .5,annot_kws = {"size": 11},mask = mask)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right')
plt.show()

print("Features which need to be encoded are : \n" ,categorical_features)

x = df.drop(['label'], axis=1)
y = df['label']
print(y.value_counts())

ms = MinMaxScaler()
x = ms.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
precision = precision_score(y_test, predictions, average='binary')
recall = recall_score(y_test, predictions, average='binary')
f_score = f1_score(y_test, predictions)
print(accuracy_score(y_test, predictions),precision,recall,f_score)


confusion_mtx = metrics.confusion_matrix(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=model.classes_)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)

fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['DDoS', 'Норма']).plot()
plt.title('Матрица Ошибок')
plt.xlabel('Предсказанные значения')
plt.ylabel('Реальные значения')
from collections import Counter

plt.show()


feature_importance = model.feature_importances_
i = list(df.columns)
i.remove('label')
feature_names = np.array(i)
data={'feature_names':feature_names,'feature_importance':feature_importance}
fi_df = pd.DataFrame(data)
print(feature_importance,feature_names,data)

fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
fi_df = fi_df.head(3)
#Define size of bar plot
plt.figure(figsize=(10,8))
#Plot Searborn bar chart
sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#Add chart labels
plt.title('Важность атрибутов')
plt.xlabel('Важность атрибута')
plt.ylabel('Имя атрибута')


plt.show()


# plt.show()