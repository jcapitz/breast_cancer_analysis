# required module imports
import numpy as np
import pandas as pd

# dataparser function
def datadesigner(datafile):
    '''
    This function takes data in the ICOS data format
    and rearranges it for analysis
    PARAMETERS: file path. Example: ~/Documents/foo.txt
    RETURN: X, the design matrix and Y, the response vector, gene names
    '''
    data = pd.read_table(datafile, delim_whitespace = True, dtype={'a': np.float64}, header = None)

    # extract colum names (all but last one are gene names)
    colNames = data[0]
    colNames = list(colNames)

    # tranpose the data to have genes as columns and microarray experiment as rows
    data = data.T.ix[1:].copy()

    # reset the index to start at 0
    data = data.reset_index(drop=True)

    # add colum labels
    data.columns=colNames
    
    # create date set with genes only. This is the features matrix
    X = data[range(data.shape[1]-1)]
    # make sure data type is float and not string.
    X = X.astype(float)
    
    # create vector with response variable
    Y = data[[-1]]
    # make sure response vector is binary 0, 1.
    Y = pd.get_dummies(Y)
    # choose one of the equivalent response vectors.
    Y = Y.ix[:,0]

    geneNames = X.columns

    return X,Y,geneNames

def geneSelector(X,Y,geneNames,M):
    '''
    Description: This function performs feature selection based on
    Elastic Net regularized logistic regression.
    INPUT:
    OUTPUT:
    '''
    # imports
    from sklearn.linear_model import SGDClassifier
    from sklearn.cross_validation import train_test_split
    
    # set up elastic net model
    model = SGDClassifier(loss = 'log', penalty = 'elasticnet', alpha = 0.175, 
                                l1_ratio = 0.5, fit_intercept = True)
    
    #initialize the dataframes for ranking genes
    selected_genes = {'col1':'gene1'}
    gene_names = pd.DataFrame(geneNames)
    gene_names.columns = ['trial0']

    # Initial trial column 
    gene_select_count = gene_names.isin(selected_genes)
    gene_coeff = pd.DataFrame()

    # IN THE LOOP
    for i in range(M):
        # Split the data

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        # Fit model and select significant genes

        fit_model = model.fit(X_train, Y_train)
        X_selected = fit_model.transform(X_test)

        # Find the indexes of significant genes

        # these are the index of the features selected by the l1 regularized logistic model
        selected_index = np.where(model.coef_!=0)[-1]
        # Selected genes in ith trial
        selected_genes = gene_names.loc[selected_index]

        gene_select_count['trial' + str(i)] = gene_names.isin(selected_genes)
        gene_coeff['trial' + str(i)] = pd.Series(model.coef_[0])
        
    return {'coeff':gene_coeff,'frequency':gene_select_count}
