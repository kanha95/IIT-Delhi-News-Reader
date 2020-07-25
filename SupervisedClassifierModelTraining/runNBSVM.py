import ktrain
from sklearn.metrics import confusion_matrix
from ktrain import text
import pickle
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit



DATA_PATH = 'Labelled_Chunks/trainV2.csv'

data = pd.read_csv(DATA_PATH, encoding='utf-8', sep=',')
X = data['sentence'].values.tolist()
Y = data['label'].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_idx, test_idx in sss.split(X, Y):
    #print(train_idx, test_idx)
    xtrain = [X[i] for i in train_idx] 
    ytrain = [Y[i] for i in train_idx] 
    xtest = [X[i] for i in test_idx] 
    ytest = [Y[i] for i in test_idx] 


xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xtest = np.array(xtest)
ytest = np.array(ytest)

#for i in range(5,6):


MAXLEN = 512 #can give anything

(x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=xtrain, y_train=ytrain,
                                        x_test=xtest, y_test=ytest,
                                        class_names=['0', '1', '2'],
                                        max_features=1000,
                                        ngram_range=2,
                      			maxlen=MAXLEN)

model = text.text_classifier('nbsvm', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))

learner.load_model('Unsupervised_Models/modelNBSVM')
#learner.lr_find()
#learner.lr_plot()

learner.autofit(0.001)

#learner.fit(2e-5, 4)
#learner.fit(2e-5, 6, cycle_len=1, cycle_mult=2)

learner.save_model('Unsupervised_Models/modelNBSVM')

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('Prediction_Classifiers/predictorNBSVM')

#y_pred = np.array(predictor.predict(xtest), dtype='int8')
#print(ytest)
#print(y_pred)
#cm = confusion_matrix(ytest, y_pred, labels=[0, 1, 2])

#for i in range(len(y_pred)):
 #   if ytest[i] != y_pred[i]:
  #      print(xtest[i], ytest[i], y_pred[i])

#print(cm)

