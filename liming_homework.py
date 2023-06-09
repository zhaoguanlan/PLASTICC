import sklearn.metrics


def contingency_table(prediction_condition,true_condition):
    a = len(prediction_condition)
    b = len(true_condition)

    if (a!=b):
        raise Exception("Different lengths")
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(prediction_condition)):
        if (prediction_condition[i]+true_condition[i]==2):
            tp=tp+1
        elif (prediction_condition[i]+true_condition[i]==0):
            tn=tn+1
        elif (prediction_condition[i]==0 and true_condition[i] == 1):
            fn= fn+1
        elif (prediction_condition[i]==1 and true_condition[i] == 0):
            fp=fp+1
    return [[tp,fn],[fp,tn]]

def calc_sens_spec(contab):
    if(contab[0][0]!=0):
        sens = float(contab[0][0] / (contab[0][0]+contab[0][1]))
    else:
        sens = 0
    if(contab[1][1]!=0):
        spec = float(contab[1][1] / (contab[1][1] + contab[1][0]))
    else:
        spec = 0
    return (sens,spec)

#_________________________________________________

class Cafe:
    def __init__(self):
        self.money = 0
        self.cake_in_stock = 10
        self.sell_price = 3
        self.buy_price = 1
    def buy_cake(self,pieces_of_cake):

        if (pieces_of_cake * self.buy_price > self.money):
            raise Exception("we do not have enough money")
        self.money = self.money - pieces_of_cake *self.buy_price
        self.cake_in_stock = self.cake_in_stock +pieces_of_cake

    def sell_cake(self,pieces_of_cake):

        if (pieces_of_cake > self.cake_in_stock):
            raise Exception("we do not have enough cake")
        self.money = self.money + pieces_of_cake*self.sell_price
        self.cake_in_stock = self.cake_in_stock - pieces_of_cake

class RescueCentre:
    def __init__(self,input):
        self.animals = input
        self.adopted_animals = {}
    def rescue(self,new_pets):
        for key in new_pets.keys():
            if key in self.animals.keys():
                self.animals[key].append(new_pets[key][0])
            else:
                self.animals[key]=new_pets[key]

    def adopt(self,animal_type):

        if animal_type in self.animals.keys():
            if animal_type in self.adopted_animals.keys():
                animal = self.animals[animal_type].pop(0)
                self.adopted_animals[animal_type].append(animal)
            else:
                self.adopted_animals[animal_type] = [self.animals[animal_type].pop(0)]
            if (self.animals[animal_type]==[]):
                self.animals.pop(animal_type)
        else:
            raise Exception("No such animals")
"""
-------------------------------------------------
#  ( a )
import pandas as pd
whiskies = pd.read_csv("https://raw.githubusercontent.com/UofGAnalyticsData/DPIP/main/whiskies.csv")
import matplotlib.pyplot as plt

lat_range = max(whiskies['Latitude']) - min(whiskies['Latitude'])
long_range = max(whiskies['Longitude']) - min(whiskies['Longitude'])
plt.figure(figsize=(lat_range/50000,long_range/50000))

plt.title("fig1")
plt.xlabel("latitude")
plt.ylabel(" lonoitude")

plt.scatter(whiskies['Latitude']/50000, whiskies['Longitude']/50000, edgecolors='none', s=20)
plt.show()

#  ( b )

col = []
dic = []
colors = ['red','orange','yellow','green','purple']

for i in range(len(whiskies['Region'])):
    if whiskies['Region'][i] not in dic:
        dic.append(whiskies['Region'][i])
#print(dic)
#['Highland', 'Speyside', 'Islay', 'Lowland', 'Campbeltown']

freq = [0,0,0,0,0]

for i in range(len(whiskies['Region'])):
    index = dic.index(whiskies['Region'][i])
    col.append(colors[index])
    freq[index]+=1

plt.title("fig2")
plt.xlabel("latitude")
plt.ylabel(" lonoitude")

plt.scatter(whiskies['Latitude']/50000, whiskies['Longitude']/50000, c = col,edgecolors='none', s=20)
plt.show()

#  ( c )

plt.title("fig3")
plt.pie(freq,labels=dic,shadow=True,colors=colors)
plt.show()

"""

"""
#(a)
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
dataset = load_breast_cancer()
import numpy as np
from sklearn import preprocessing
import sklearn.model_selection

X = dataset.data
y = dataset.target

mean = np.mean(X,axis = 0)
std = np.std(X, axis = 0)

#(b)

X_ = preprocessing.StandardScaler().fit_transform(X)

#(c)

train_X,test_X,train_y,test_y = sklearn.model_selection.train_test_split(X_,y,test_size=0.2,random_state=1)

#(d)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
# Create an instance of Logistic Regression Classifier and fit the data.
logreg.fit(train_X, train_y)
y_pred = logreg.predict(test_X)
y_pred_train = logreg.predict(train_X)

#(e)

print("Accuracy and F1 scores on test datasets",sklearn.metrics.classification_report(test_y,y_pred))
print("Accuracy and F1 scores on the training datasets",sklearn.metrics.classification_report(train_y,y_pred_train))

#(f)

from sklearn.feature_selection import GenericUnivariateSelect
import sklearn.feature_selection
score_func = sklearn.feature_selection.mutual_info_classif
transformer = GenericUnivariateSelect(score_func, mode='k_best', param=15)
X_updata = transformer.fit_transform(X, y)
score = score_func(X, y)

x_data = np.argsort(score)
y_data = score
for i in range(15):
    plt.bar(x_data[29-i],y_data[x_data[29-i]])
plt.title("top 15")
plt.xlabel("covariates")
plt.ylabel("scores")
plt.show()

#(g)

X_updata = preprocessing.StandardScaler().fit_transform(X_updata)
train_X,test_X,train_y,test_y = sklearn.model_selection.train_test_split(X_updata,y,test_size=0.2,random_state=1)
logreg_updata = LogisticRegression()
# Create an instance of Logistic Regression Classifier and fit the data.
logreg_updata.fit(train_X, train_y)
y_pred = logreg_updata.predict(test_X)
y_pred_train = logreg_updata.predict(train_X)

print("X_updata----Accuracy and F1 scores on test datasets",sklearn.metrics.classification_report(test_y,y_pred))
print("X_updata----Accuracy and F1 scores on the training datasets",sklearn.metrics.classification_report(train_y,y_pred_train))

"""
#(a)
from scipy import stats
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rv_norm1 = stats.norm.rvs(loc = 1,scale = 1,size =(1000))
rv_norm2 = stats.norm.rvs(loc = 3,scale = 1,size =(1000))
rv_norm3 = stats.norm.rvs(loc = 3,scale = 1,size =(1000))
#(b)
df1 = pd.DataFrame()

df1['x1'] = rv_norm1
df1['x2'] = rv_norm2
df1['x3'] = rv_norm3
#(c)
mean = np.mean(df1,axis = 0)
std = np.std(df1, axis = 0)
summary = pd.DataFrame()
summary['mean'] = mean
summary['std'] = std
print(summary)

#(d)

levene1 = stats.levene(rv_norm1, rv_norm2, center='median')
print('w-value=%6.4f,p-value=%6.4f' % levene1)
print(stats.ttest_ind(rv_norm1, rv_norm2, equal_var=True))

levene2 = stats.levene(rv_norm2, rv_norm3, center='median')
print('w-value=%6.4f,p-value=%6.4f' % levene2)
print(stats.ttest_ind(rv_norm1, rv_norm2, equal_var=True))

levene3 = stats.levene(rv_norm1, rv_norm3, center='median')
print('w-value=%6.4f,p-value=%6.4f' % levene3)
print(stats.ttest_ind(rv_norm1, rv_norm2, equal_var=True))
#(e)

import seaborn as sns
sns.kdeplot(df1['x1'])
sns.kdeplot(df1['x2'])
sns.kdeplot(df1['x3'])
plt.legend(["x1", "x2", "x3"])

plt.xlabel("normal distribution")
plt.ylabel("Density")
plt.show()