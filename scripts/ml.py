import random
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


from processing import limit_features, balance_sample, split_and_feature_select



class UpsampleStratifiedKFold:
    '''Class for passing to GridSearchCV to separately upsample training
    and validation data during the CV process (avoids data leakage)'''
    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    #Perform a stratified KFold split, then upsample training and testing data separately
    def split(self, X, y, groups=None):
        for train_ix, test_ix in StratifiedKFold(n_splits=self.n_splits).split(X,y):
            
            #upsample training
            neg_ix = np.where(y[train_ix]==0)[0]
            pos_ix = np.where(y[train_ix]==1)[0]
            pos_ix_upsampled = np.random.choice(pos_ix, size=neg_ix.shape[0], replace=True)
            assert(len(pos_ix_upsampled) == len(neg_ix))
            ix = np.append(neg_ix, pos_ix_upsampled)
            train_ix = train_ix[ix]
            
            
            #upsample testing
            neg_ix = np.where(y[test_ix]==0)[0]
            pos_ix = np.where(y[test_ix]==1)[0]
            pos_ix_upsampled = np.random.choice(pos_ix, size=neg_ix.shape[0], replace=True)
            assert(len(pos_ix_upsampled) == len(neg_ix))
            ix = np.append(neg_ix, pos_ix_upsampled)
            test_ix = test_ix[ix]
            
            yield train_ix, test_ix

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def experiment_with_models(X_train, X_test, y_train, y_test):
    '''Takes pre split training and testing data, and tests these data against numerous models.
    Models are evaluated using CV, where both train/ validation sets are oversampled during CV.
    Returns the names of model archetypes evaluated and their testing accuracies (for their best hyperparams). '''
    #make sure behavior is the same 'random' across time
    np.random.seed(0)
    
    #keep track of the models I ran and their best performances
    model_names = []
    best_model_accs = []
    
    #Determine the naive binary classification accuracy (should be 0.5 for balanced data)
    #FOR THE TESTING DATA
    num_tbi = np.sum(y_test)
    num_hc = len(y_test) - num_tbi
    majority_class_len = max(num_tbi, num_hc)
    naive_accuracy = majority_class_len/(num_tbi + num_hc)
    print(f"Expecting a naive TESTING accuracy of {naive_accuracy}")
    print(f"Note: Validation Data is Balanced Separately Among Train/Val split data...")
    print(f"Therefore, validation naive accuracy = 0.5")
    print()
    
    #KNN
    print("---------------------KNN----------------------")
    model_names.append('KNN')
    knn = KNeighborsClassifier()
    params = {'n_neighbors': range(1, 20), 
              'weights': ['uniform', 'distance']
             }
    clf = GridSearchCV(estimator = knn, param_grid = params, error_score = 'raise', cv =UpsampleStratifiedKFold(n_splits=10))
    clf.fit(X_train, y_train)

    best_knn = clf.best_estimator_    
    print(f"Best KNN Model: {best_knn}")
    print(f"Best KNN CV Validation Accuracy:{clf.best_score_}")
    preds = best_knn.predict(X_test)
    print(f"Best KNN (Single Run) Testing Accuracy: {accuracy_score(preds, y_test)}")
    best_model_accs.append(accuracy_score(preds, y_test))
    print("----------------------------------------------")
    print()

    ## Decision Tree
    print("--------------------DT-------------------------")
    model_names.append('DT')
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    print(f"Basic DT Test Accuracy: {accuracy_score(dt.predict(X_test), y_test)}")
    
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': range(1, 30, 2),
              'max_features': ['sqrt', 'log2']
    }
    clf = GridSearchCV(estimator = dt, param_grid = params, cv =UpsampleStratifiedKFold(n_splits=10))
    clf.fit(X_train, y_train)
    best_dt = clf.best_estimator_
    print(f'Best Decision Tree: {best_dt}')
    print(f'Validation Accuracy: {clf.best_score_}')
    print(f'Test Accuracy: {accuracy_score(best_dt.predict(X_test), y_test)}')
    best_model_accs.append(accuracy_score(best_dt.predict(X_test), y_test))
    print("----------------------------------------------")
    print()
          
    ## Support Vector Machine
    print("--------------------SVC------------------------")
    model_names.append('SVC')
    svc = SVC(kernel = 'rbf', gamma = 'scale')
    svc.fit(X_train, y_train)
    print(f"Basic (RBF, Scale) SVC Accuracy: {accuracy_score(svc.predict(X_test), y_test)}")

    gamma_params = ['scale', 'auto']
    #can unhash the following for more choices on gamma (slow):
    #gamma_params.extend(list(range(1,30)))
    params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': range(1, 10,1),
              'gamma': gamma_params,
              'shrinking': [False, True] 
    }
    clf = GridSearchCV(estimator = svc, param_grid = params, cv =UpsampleStratifiedKFold(n_splits=10))
    #NOTE: THIS GRIDSEARCH IS PRETTY SLOW
    clf.fit(X_train, y_train)
    best_svc = clf.best_estimator_
    print(f"Best SVC: {best_svc}")
    print(f"Best SVC Validation Accuracy: {clf.best_score_}")
    print(f"Best SVC Test Accuracy: {accuracy_score(best_svc.predict(X_test), y_test)}")
    best_model_accs.append(accuracy_score(best_svc.predict(X_test), y_test))
    print("----------------------------------------------")
    print()
    
    ##Logistic Regression
    print("--------------------LogReg--------------------")
    model_names.append('LogReg')
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print(f"Basic LogReg Test Accuracy: {accuracy_score(logreg.predict(X_test), y_test)}")
    params = {'penalty' : ['l1', 'l2'],
              'max_iter' : [10000],
              'solver': ['liblinear'],
              'C': np.arange(0.5, 5.5, 0.5)
        
    }
    clf = GridSearchCV(estimator = logreg, param_grid = params, cv=UpsampleStratifiedKFold(n_splits=10))
    clf.fit(X_train, y_train)
    best_logreg = clf.best_estimator_
    print(f"Best Logistic Regression: {best_logreg}")
    print(f"Best LogReg Validation Accuracy: {clf.best_score_}")
    print(f"Best LogReg Test Accuracy: {accuracy_score(best_logreg.predict(X_test), y_test)}")
    best_model_accs.append(accuracy_score(best_logreg.predict(X_test), y_test))
    print("----------------------------------------------")
    print()
    
    ##Naive Bayes
    print("-----------------Naive Bayes------------------")
    model_names.append('Guassian NB')
    #using gaussian bayes, so that we use mean and standard deviation of continuous features to calculate probs
    #note that this assumes independence between all of the features I've selected
    #which is in reality, unlikely
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    print(f"Gaussian Naive Bayes Test Accuracy: {accuracy_score(gnb.predict(X_test), y_test)}")
    best_model_accs.append(accuracy_score(gnb.predict(X_test), y_test))
    
    #not rlly parameters to tune with naive bayes
    print("----------------------------------------------")
    print()
    
    ##Random Forest
    print("-----------------Random Forest----------------")
    model_names.append('RF')
    #Note: Random Forest models are definitely slower to train, so the GridSearch is pretty slow here
    rf = RandomForestClassifier(random_state = 42)
    rf.fit(X_train, y_train)
    print(f"Basic RandomForest Test Accuracy: {accuracy_score(rf.predict(X_test), y_test)}")
    params = {'criterion': ['gini', 'entropy'],
              'max_depth': range(25, 45, 2),
              'max_features': ['sqrt', 'log2'],
              'bootstrap': [False, True],
              'random_state': [42]
    }
    clf = GridSearchCV(estimator = rf, param_grid = params, cv = UpsampleStratifiedKFold(n_splits=10))
    clf.fit(X_train, y_train)
    best_rf = clf.best_estimator_
    print(f'Best RF Model: {best_rf}')
    print(f'Validation Accuracy: {clf.best_score_}')
    print(f'Test Accuracy: {accuracy_score(best_rf.predict(X_test), y_test)}')
    best_model_accs.append(accuracy_score(best_rf.predict(X_test), y_test))
    
    print("----------------------------------------------")
    print()
    print('Done.')
    
    return model_names, best_model_accs



def plot_ml_results(model_names, accs, y_test): 
    '''Takes in a list of model names and test accuracies, as well as the y_test var.
    Plots results in a barplot for visualization w/ axhline to visualize naive accuracy based
    on unbalanced data. '''
    #Determine the naive binary classification accuracy (should be 0.5 for balanced data)
    #FOR THE TESTING DATA
    num_tbi = np.sum(y_test)
    num_hc = len(y_test) - num_tbi
    majority_class_len = max(num_tbi, num_hc)
    naive_accuracy = majority_class_len/(num_tbi + num_hc)

    plt.figure(figsize=(8,8))
    plt.bar(model_names, accs, alpha = 0.8)
    plt.suptitle(f'Testing Accuracy of {len(model_names)} ML Model Archetypes')
    plt.title('(Each Model was Tuned for Optimal Hyperparams)')
    plt.xlabel('Model')
    plt.ylabel(f'Accuracy (Naive accuracy = {naive_accuracy})')
    plt.axhline(y=naive_accuracy, color = 'green', label = 'Naive Accuracy')
    plt.legend(loc = 3)
    plt.show()
    
    return