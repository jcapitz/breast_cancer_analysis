#!/usr/bin/python
def import_mods():
    # Data Imports
    import numpy as np
    import pandas as pd
    from pandas import Series,DataFrame

    import scipy.cluster.hierarchy as hac
    from scipy.stats import norm


    # Math
    import math

    # Plot imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')

    # Preprocessing Imports
    from sklearn.preprocessing import normalize

    # Machine Learning Imports
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier


    from sklearn.linear_model import RandomizedLogisticRegression
    from sklearn.cross_validation import train_test_split

    # For evaluating our ML results
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score, confusion_matrix

    # For Spectral Clustering algorithms
    from sklearn import cluster
    from sklearn.neighbors import kneighbors_graph

    # network imports
    import networkx as nx
    import igraph as ig
