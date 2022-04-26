# This file contains model paramters of baselines models

# for topography images
model_params = {
'NN': [{'activation': 'relu','alpha': 0.0001,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'solver': 'adam',}],
'SVC_linear': [{'C': 0.1}],
'SVC_rbf': [{'C': 10,'gamma': 0.01,'kernel': 'rbf',}],
'SVC_poly': [{'C': 0.01,'degree': 3,'gamma': 0.01,'kernel': 'poly',}],
}

model_params_pca = {
'NN': [{'activation': 'tanh','alpha': 0.0001,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'solver': 'adam',}],
'SVC_linear': [{'C': 1}],
'SVC_rbf': [{'C': 10,'gamma': 0.01,'kernel': 'rbf',}],
'SVC_poly': [{'C': 0.01,'degree': 3,'gamma': 0.01,'kernel': 'poly',}],
}

# for deflection images
model_params = {
'NN': [{'activation': 'relu','alpha': 0.0001,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'solver': 'adam',}],
'SVC_linear': [{'C': 0.01}],
'SVC_rbf': [{'C': 10,'gamma': 0.01,'kernel': 'rbf',}],
'SVC_poly': [{'C': 0.01,'degree': 3,'gamma': 0.01,'kernel': 'poly',}],
}

model_params_pca = {
'NN': [{'activation': 'relu','alpha': 0.9,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'solver': 'adam',}],
'SVC_linear': [{'C':0.01}],
'SVC_rbf': [{'C': 10,'gamma': 0.01,'kernel': 'rbf',}],
'SVC_poly': [{'C': 0.01,'degree': 3,'gamma': 0.01,'kernel': 'poly',}],
}

