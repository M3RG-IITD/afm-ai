from joblib import Parallel, delayed
import multiprocessing
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier as MLPC

from sklearn.metrics import r2_score, f1_score, classification_report, confusion_matrix, recall_score, precision_score

from tqdm import tqdm


num_cores = multiprocessing.cpu_count()

model = SVC(kernel='linear')

class custom_KFold:

    # instance attribute
    def __init__(self, model=SVC(kernel='linear'), kfolds=5,x=None,y=None,name='SVC'):
        self.model = model
        self.models = []
        self.folds = kfolds
        self.x = x
        self.y = y
        
        global models
        models = []
#         self.scores = None
        x_train, x_test, y_train, y_test = train_test_split(self.x,self.y,test_size = 0.15,random_state=43)
        try:
            if self.model.kernel =='linear':
                name = 'svc_linear'
                C = [0.01,0.1,1,10,100]
                kf = KFold(n_splits=self.folds)
                resource_list = []
                idx = 0
                for c in C:
                    for train_index, val_index in kf.split(x_train):
                        idx = idx + 1
                        resource_list.append([idx,c,train_index,val_index])

                inputs = tqdm(resource_list)
                model = SVC(kernel='linear')
                # if __name__ == "__main__":
                processed_list = Parallel(n_jobs=num_cores)(delayed(fireSVC)(i,model,x_train,y_train,name) for i in inputs)
                self.scores = processed_list
    #             self.models = processed_list[0]
            if self.model.kernel =='rbf':
                name = 'svc_rbf'
                C = [0.01,0.1,1,10,100]
                G = [0.01,0.1,1,10,100]

                kf = KFold(n_splits=self.folds)
                resource_list = []
                idx = 0
                for c in C:
                    for g in G:
                        for train_index, val_index in kf.split(x_train):
                            idx = idx + 1
                            resource_list.append([idx,c,g,train_index,val_index])


                inputs = tqdm(resource_list)
                model = SVC(kernel='rbf')
                # if __name__ == "__main__":
                processed_list = Parallel(n_jobs=num_cores)(delayed(fireSVC)(i,model,x_train,y_train,name) for i in inputs)
                self.scores = processed_list
    #             self.models = processed_list[0]

            if self.model.kernel =='poly':
                name = 'svc_poly'

                degrees = [3]#,4]#[1,2,3,4,5,6,7]#,0.1]#,1,10,100]
                C = [0.01,0.1,1,10,100]
                G = [0.01,0.1,1,10,100]

                kf = KFold(n_splits=self.folds)
                resource_list = []
                idx = 0

                for d in degrees:
                    for c in C:
                        for g in G:
                            for train_index, val_index in kf.split(x_train):
                                idx = idx + 1
                                resource_list.append([idx,d,c,g,train_index,val_index])

                inputs = tqdm(resource_list)
                model = SVC(kernel='poly')
                # if __name__ == "__main__":
                processed_list = Parallel(n_jobs=num_cores)(delayed(fireSVC)(i,model,x_train,y_train,name) for i in inputs)
                self.scores = processed_list
        except:
            if str(type(self.model)) == "<class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>":
                name = 'nn_mlpc'
                activations = ['tanh','relu']
                solvers = ['sgd','adam']
                alphas = [0.0001,0.001,0.01,0.1,0.9]
                self.model.learning_rate ='adaptive'
                self.model.hidden_layer_sizes = (128,64)
                kf = KFold(n_splits=self.folds)
                resource_list = []
                idx = 0

                for activation in activations:
                    for solver in solvers:
                        for alpha in alphas:
                            for train_index, val_index in kf.split(x_train):
                                idx = idx + 1
                                resource_list.append([idx,activation,solver,alpha,train_index,val_index])
                inputs = tqdm(resource_list)
                model = self.model
                # if __name__ == "__main__":
                processed_list = Parallel(n_jobs=num_cores)(delayed(fireSVC)(i,model,x_train,y_train,name) for i in inputs)
                self.scores = processed_list
            
            
            
def fireSVC(rsc,model,x_train,y_train,name,save=False):
    models = []
    try:
        if model.kernel == 'linear':
            k = rsc[0]
            c = rsc[1]
            train_index = rsc[2]
            val_index = rsc[3]

            train_x, val_x = x_train[train_index], x_train[val_index]
            train_y, val_y = y_train[train_index], y_train[val_index]
            model.C = c
            model = model.fit(train_x, train_y)
            if save:
                with open(name+'_'+str(k).zfill(2)+'.pickle','wb') as f:
                    pickle.dump(model,f)
            models.append(model)
            scores_vl = conf_mat(val_y,model.predict(val_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn = conf_mat(train_y,model.predict(train_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn.extend(scores_vl)
            output = [models,scores_tn]
        if model.kernel == 'rbf':
            k = rsc[0]
            c = rsc[1]
            g = rsc[2]
            train_index = rsc[3]
            val_index = rsc[4]

            train_x, val_x = x_train[train_index], x_train[val_index]
            train_y, val_y = y_train[train_index], y_train[val_index]
            model.C = c
            model.gamma = g
            model = model.fit(train_x, train_y)
            if save:
                with open(name+'_'+str(k).zfill(3)+'.pickle','wb') as f:
                    pickle.dump(model,f)
            models.append(model)
            scores_vl = conf_mat(val_y,model.predict(val_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn = conf_mat(train_y,model.predict(train_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn.extend(scores_vl)
            output = [models,scores_tn]

        if model.kernel == 'poly':
            k = rsc[0]
            d = rsc[1]
            c = rsc[2]
            g = rsc[3]

            train_index = rsc[4]
            val_index = rsc[5]

            train_x, val_x = x_train[train_index], x_train[val_index]
            train_y, val_y = y_train[train_index], y_train[val_index]
            model.degree = d
            model.C = c
            model.gamma = g
            model = model.fit(train_x, train_y)
            if save:
                with open(name+'_'+str(model.degree)+'_'+str(k).zfill(2)+'.pickle','wb') as f:
                    pickle.dump(model,f)
            models.append(model)
            scores_vl = conf_mat(val_y,model.predict(val_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn = conf_mat(train_y,model.predict(train_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn.extend(scores_vl)
            output = [models,scores_tn]
    except:
        if str(type(model)) == "<class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>":
            name = 'MLPC'
            k = rsc[0]
            activation = rsc[1]
            solver = rsc[2]
            alpha = rsc[3]

            train_index = rsc[4]
            val_index = rsc[5]

            train_x, val_x = x_train[train_index], x_train[val_index]
            train_y, val_y = y_train[train_index], y_train[val_index]
            model.activation = activation
            model.solver = solver
            model.alpha = alpha

            model = model.fit(train_x, train_y)
            if save:
                with open(name+'_'+str(model.degree)+'_'+str(k).zfill(2)+'.pickle','wb') as f:
                    pickle.dump(model,f)
            models.append(model)
            scores_vl = conf_mat(val_y,model.predict(val_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn = conf_mat(train_y,model.predict(train_x),name+'_'+str(k).zfill(2),save=False)
            scores_tn.extend(scores_vl)
            output = [models,scores_tn]
    
    
    return output 


        
def conf_mat(y_true,y_pred,name,save=False):
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()

    labels = ['TrueNeg='+str(tn),'FalsePos='+str(fp),'FalseNeg='+str(fn),'TruePos='+str(tp)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix(y_true,y_pred),annot=labels, fmt='', cmap='Blues')
#     plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    ps = precision_score(y_true, y_pred)
    rs = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    title= "F1_Score= "+str(round(f1_score(y_true, y_pred),3)) + '\n' + "Recall= "+str(round(recall_score(y_true, y_pred),3)) + '\n' + "Precision= "+str(round(precision_score(y_true, y_pred),3))
    plt.title(title)
    if save:
        plt.savefig(str(name)+'.png',bbox_inches='tight')
    
    plt.show()
#     print(ps,rs,f1)
    return [ps,rs,f1]

def gen_data(img):
    #0
    transform = []
    transform.append(img.flatten())
    #1
    img1 = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    transform.append(img1.flatten())
    #2
    img2 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    transform.append(img2.flatten())
    #3
    img3 = cv2.rotate(img2,cv2.ROTATE_90_CLOCKWISE)
    transform.append(img3.flatten())
    
    #4
    h0   = np.fliplr(img)  #fliplr reverse the order of columns of pixels in matrix
    transform.append(h0.flatten())
    #5
    h1   = np.fliplr(img1) #fliplr reverse the order of columns of pixels in matrix
    transform.append(h1.flatten())
    #6
    h2   = np.fliplr(img2) #fliplr reverse the order of columns of pixels in matrix
    transform.append(h2.flatten())
    #7
    h3   = np.fliplr(img3) #fliplr reverse the order of columns of pixels in matrix
    transform.append(h3.flatten())
    
    return transform
