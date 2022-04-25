# This file contains model paramters of baselines models

# for topography images
model_params = {
'NN': [{'activation': 'relu','alpha': 0.0001,'batch_size': 'auto','beta_1': 0.9,'beta_2': 0.999,'early_stopping': False,'epsilon': 1e-08,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'max_fun': 15000,'max_iter': 200,'momentum': 0.9,'n_iter_no_change': 10,'nesterovs_momentum': True,'power_t': 0.5,'random_state': None,'shuffle': True,'solver': 'adam','tol': 0.0001,'validation_fraction': 0.1,'verbose': False,'warm_start': False}],
'SVC_linear': [{'C': 0.1,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 'scale','kernel': 'linear','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_rbf': [{'C': 10,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'rbf','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_poly': [{'C': 0.01,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'poly','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
}

model_params_pca = {
'NN': [{'activation': 'tanh','alpha': 0.0001,'batch_size': 'auto','beta_1': 0.9,'beta_2': 0.999,'early_stopping': False,'epsilon': 1e-08,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'max_fun': 15000,'max_iter': 200,'momentum': 0.9,'n_iter_no_change': 10,'nesterovs_momentum': True,'power_t': 0.5,'random_state': None,'shuffle': True,'solver': 'adam','tol': 0.0001,'validation_fraction': 0.1,'verbose': False,'warm_start': False}],
'SVC_linear': [{'C': 1,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 'scale','kernel': 'linear','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_rbf': [{'C': 10,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'rbf','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_poly': [{'C': 0.01,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'poly','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
}

# for deflection images
model_params = {
'NN': [{'activation': 'relu','alpha': 0.0001,'batch_size': 'auto','beta_1': 0.9,'beta_2': 0.999,'early_stopping': False,'epsilon': 1e-08,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'max_fun': 15000,'max_iter': 200,'momentum': 0.9,'n_iter_no_change': 10,'nesterovs_momentum': True,'power_t': 0.5,'random_state': None,'shuffle': True,'solver': 'adam','tol': 0.0001,'validation_fraction': 0.1,'verbose': False,'warm_start': False}],
'SVC_linear': [{'C': 0.01,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 'scale','kernel': 'linear','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_rbf': [{'C': 10,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'rbf','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_poly': [{'C': 0.01,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'poly','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
}

model_params_pca = {
'NN': [{'activation': 'relu','alpha': 0.9,'batch_size': 'auto','beta_1': 0.9,'beta_2': 0.999,'early_stopping': False,'epsilon': 1e-08,'hidden_layer_sizes': (128, 64),'learning_rate': 'adaptive','learning_rate_init': 0.001,'max_fun': 15000,'max_iter': 200,'momentum': 0.9,'n_iter_no_change': 10,'nesterovs_momentum': True,'power_t': 0.5,'random_state': None,'shuffle': True,'solver': 'adam','tol': 0.0001,'validation_fraction': 0.1,'verbose': False,'warm_start': False}],
'SVC_linear': [{'C':0.01,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 'scale','kernel': 'linear','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_rbf': [{'C': 10,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'rbf','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
'SVC_poly': [{'C': 0.01,'break_ties': False,'cache_size': 200,'class_weight': None,'coef0': 0.0,'decision_function_shape': 'ovr','degree': 3,'gamma': 0.01,'kernel': 'poly','max_iter': -1,'probability': False,'random_state': None,'shrinking': True,'tol': 0.001,'verbose': False}],
}

