from encoder import Model
from matplotlib import pyplot as plt
from utils import sst_binary, train_with_reg_cv
import numpy as np
import os
from sklearn import svm,metrics
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Model('./model/0/model.npy')

trX, vaX, teX, trY, vaY, teY = sst_binary()

if not os.path.exists('features/labelleddata'):
    os.makedirs('features/labelleddata')

    trXt = model.transform(trX)
    vaXt = model.transform(vaX)
    teXt = model.transform(teX)
    print(trXt.shape)

    np.save('features/labelleddata/trXt',trXt)
    np.save('features/labelleddata/vaXt',vaXt)
    np.save('features/labelleddata/teXt',teXt)

else:
    print('load features')
    trXt = np.load('features/labelleddata/trXt.npy')
    vaXt = np.load('features/labelleddata/vaXt.npy')
    teXt = np.load('features/labelleddata/teXt.npy')

print(trXt.shape, trY.shape)
"""
#NEURAL NETWORK

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trXt = sc.fit_transform(trXt)
vaXt = sc.fit_transform(vaXt)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
trY = ohe.fit_transform(np.array(trY).reshape(-1,1)).toarray()
vaY = ohe.fit_transform(np.array(vaY).reshape(-1,1)).toarray()

model = Sequential()
model.add(Dense(units=30, activation='relu', input_dim=4096))
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(trXt).reshape(-1,4096), trY, epochs=50, batch_size=32)
loss_and_metrics = model.evaluate(np.array(vaXt).reshape(-1,4096), vaY, batch_size=32)

print(loss_and_metrics)


"""


"""
#XGBOOST

model = XGBClassifier()
model.fit(trXt, trY)

Ypred = model.predict(vaXt)
print("Validation Accuracy: ", metrics.accuracy_score(vaY, Ypred.flatten()))


#for i in range(len(vaY)):
 #   if vaY[i] != Ypred[i]:
  #      print(vaX[i], vaY[i], Ypred[i])    

#from matplotlib import pyplot
#pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()

"""





"""

#SIMPLE LOGISTIC REGRESSION CLASSIFIER

full_rep_acc, c, nnotzero, coef, lg_model = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
print('%05.2f test accuracy'%full_rep_acc)
#print('%05.2f regularization coef'%c)
#print('%05d features used'%nnotzero)
#for i,j in enumerate(coef[0]):
 #   if j>0.08:
  #      print(str(i)+" "+str(j))
"""



"""
#SVM


model = svm.SVC(kernel='rbf')
model.fit(trXt, trY)

ypred = model.predict(trXt)
print("Training Accuracy:",metrics.accuracy_score(trY, ypred))

ypred = model.predict(vaXt)
print("Validation Accuracy:",metrics.accuracy_score(vaY, ypred))

#for i,j in enumerate(model.coef_[0]):
 #   if j>0.0:
  #      print(str(i), str(j))

#for i in range(len(vaY)):
 #   if vaY[i] != ypred[i]:
  #      print(vaX[i], vaY[i], ypred[i]) 


"""
"""
#PREDICTION BASED ON NEURON VALUE
neuron = 529
value = 1.0
model = trXt[:, neuron]
y_pred = []
p = 0
for i in vaXt:
    if i[neuron] < value:
        y_pred.append(2)
    else:
        y_pred.append(0)
    print(i[neuron], y_pred[p], vaY[p])
    p = p + 1

print("Validation Accuracy:",metrics.accuracy_score(vaY, y_pred))

"""
"""
281 282 1123 1515 1601
"""


#visualize sentiment unit 529_1.0

for neuron in range(0,100):
    sentiment_unit = trXt[:, neuron]
    plt.title('Sentiment Neuron : ' + str(neuron))
    plt.xlabel('Neuron Value')
    plt.ylabel('Number of examples')
    plt.hist(sentiment_unit[trY==0], bins=25, alpha=0.5, label='neg', color='red')
    plt.hist(sentiment_unit[trY==1], bins=25, alpha=0.5, label='pos', color='green')
    #plt.hist(sentiment_unit[trY==2], bins=25, alpha=0.5, label='pos', color='green')
    plt.legend()
    plt.show()


