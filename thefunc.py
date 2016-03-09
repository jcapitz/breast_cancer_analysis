#!/usr/bin/python

# required modules
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy.stats import norm

# Machine Learning Imports
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split

# For evaluating our ML results
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix

# check for housekeeping genes
def housekeepgenes(hkfile, data):
    genenames = data[0]
    genenames = list(genenames)
    hkdata = pd.read_csv(hkfile)
    hknames = hkdata['Gene Name']
    hkgenesin = sum(hknames.isin(genenames))
    print 'there are {} house keeping genes.'.format(hkgenesin) 


def data_engineer(data,colNames):
    '''
    produce Y vector and X matrix
    this function is specific to the BCR data format
    '''
    # tranpose the data to have genes as columns and microarray experiment as rows
    data2 = data.T.ix[1:]
    # reset the index to satrt at 0
    data2 = data2.reset_index(drop=True)
    # add colum labels
    data2.columns=colNames
    # create date set with genes only. This is the features matrix
    X = data2[range(data.shape[0]-1)]
    # make sure data type is float and not string.
    X = X.astype(float)
    # create vector with response variable
    Y = data2[[-1]]
    # make sure response vector is binary 0, 1.
    Y = pd.get_dummies(Y)
    # choose one of the equivalent response vectors.
    Y = Y.ix[:,0]
    return X,Y

def geneSelector(X,Y,model,M):
    '''
    This function performs feature selection based on
    Elastic Net regularized logistic regression.
    INPUT: 
       X the feature pandas dataframe
       Y response pandas series
       model, a scikit-learn SGD linear classifier object
       M, number of iterations during which a selection occurs
    OUTPUT:
       A dictionary with the selection fequency and coefficients
       for each run
    '''
    #initialize the dataframes for ranking genes
    selected_genes = {'col1':'gene1'}
    geneNames = X.columns
    gene_names = DataFrame(geneNames)
    gene_names.columns = ['trial0']

    # Initial trial column 
    gene_select_count = gene_names.isin(selected_genes)
    gene_coeff = DataFrame()

    for i in range(M):
        # Split the data

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        # Fit model and select significant genes

        fit_model = model.fit(X_train, Y_train)

        # Find the indexes of significant genes

        # these are the index of the features selected by the l1 regularized logistic model
        selected_index = np.where(model.coef_!=0)[-1]
        # Selected genes in ith trial
        selected_genes = gene_names.loc[selected_index]

        gene_select_count['trial' + str(i)] = gene_names.isin(selected_genes)
        gene_coeff['trial' + str(i)] = Series(model.coef_[0])
        
    return {'coeff':gene_coeff,'frequency':gene_select_count}

def geneSelector2(X,Y,geneNames,model,M):
    #initialize the dataframes for ranking genes
    selected_genes = {'col1':'gene1'}
    gene_names = DataFrame(geneNames)
    gene_names.columns = ['trial0']

    # Initial trial column 
    gene_select_count = gene_names.isin(selected_genes)
    gene_impurity = DataFrame()

    for i in range(M):
        # Split the data

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        # Fit model and select significant genes

        fit_rf_model = model.fit(X, Y)

        # Find indexes of significant genes

        selected_index = np.where(model.feature_importances_!=0)[-1]

        # Selected genes in ith trial

        selected_genes = gene_names.loc[selected_index]

        gene_select_count['trial' + str(i)] = gene_names.isin(selected_genes)
        gene_impurity['trial' + str(i)] = Series(model.feature_importances_)

    return {'impurity':gene_impurity,'frequency':gene_select_count}

def penalty_selector(X,Y,k):

    alpha = np.logspace(-8,1,150)
    l1_ratio = np.arange(0.05,1.05,0.05)

    alpha_list = []
    l1_ratio_list = []
    accuracy_list = []

    for par1 in alpha:
        for par2 in l1_ratio:
            log_model = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = par1, 
                              l1_ratio = par2, fit_intercept = True)
            scores = cross_validation.cross_val_score(log_model, X, Y, cv=k)
            mean_score = scores.mean()
            alpha_list.append(par1)
            l1_ratio_list.append(par2)
            accuracy_list.append(mean_score)

    score_dict = {'alpha': alpha_list, 'l1_ratio': l1_ratio_list, 'accuracy': accuracy_list}

    return score_dict

def param_calculator(featureNames,feature_coeff,feature_select_count,threshold):
    from scipy.stats import norm
    beta_coeff = feature_coeff.mean(axis = 1)
    beta_sterror = feature_coeff.std(axis = 1)
    feature_imp_score = feature_coeff.mean(axis = 1).abs()
    pvalue = 1-norm.cdf(np.abs(beta_coeff/beta_sterror))

    # this creates a pandas Series with the sum of how many times a feature was chosen
    feature_select_summary = feature_select_count.sum(axis = 1)

    # this data frame has the values for all features
    result = DataFrame([featureNames, feature_select_summary, feature_imp_score, beta_coeff, beta_sterror, pvalue], 
                       index = ['Feature', 'Frequency', 'Score', 'Coefficient', 'Std Error', 
                               'pvalue']).T.sort(['Frequency', 'Score'], ascending = False)

    result = result[result['Score']>threshold]

    return [result,feature_select_summary]

def accuracy_calculator(model,data,response,M):

    xvalmc = []
    xvalROC = []

    for i in range(M):

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(data, response)

        # Now fit the new model
        model.fit(X_train, Y_train)

        # Predict the classes of the testing data set
        class_predict = model.predict(X_test)

        # Compare the predicted classes to the actual test classes
        xvalmc.append(metrics.accuracy_score(Y_test,class_predict))
        try:
            xvalROC.append(roc_auc_score(Y_test,class_predict))
        except ValueError:
            pass

    return [np.mean(xvalmc), np.std(xvalmc), np.mean(xvalROC), np.std(xvalROC)]

def adjecency_penalized_reg(X):
    l1_ratio_seq = np.arange(0.05,1.05,0.05)
    en_model_cv = ElasticNetCV(l1_ratio=l1_ratio_seq)
    coef_array = np.array([])
    count = 0
    
    for name in X.columns:
        en_model_cv.fit(X.drop(name, axis = 1), X[name])
        best_alpha = en_model_cv.alpha_
        best_l1_ratio = en_model_cv.l1_ratio_
        
        en_model = ElasticNet(alpha = best_alpha, l1_ratio = best_l1_ratio)
        en_model.fit(X.drop(name, axis = 1), X[name])
        
        coef_array = np.append(coef_array, np.insert(np.abs(en_model.coef_),count,0.), axis = 0)
        count += 1
    
    coef_mat = coef_array.reshape((X.shape[1],X.shape[1]))    
    return coef_mat
