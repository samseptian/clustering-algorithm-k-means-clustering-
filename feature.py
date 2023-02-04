import numpy


def FeatureSelection(dataset, method, percent):
    X = dataset['weight']['tfidf'].copy()
    vocabulary = dataset['vocabulary'].copy()
    if percent == 100:
        X_reduced = X
        vocabs = vocabulary[:]
    else:
        if method == 'pca':
            X_reduced, index = pca(X, percent)
            vocabs = [vocabulary[i] for i in index]
        else:
            DF = dataset['weight']['df'].copy()
            X_reduced, index = modified_union(X, DF, percent)
            vocabs = [vocabulary[i] for i in index]

    dataset['%s_%s' % (method, percent)] = {
        'weight': X_reduced,
        'vocabs': vocabs
    }
        
def pca(X, percent):
    n = (percent * X.shape[1]) // 100

    # calculate the mean of each column
    M = numpy.mean(X.T, axis=1)
    
    # center columns by subtracting column means
    C = X - M

    # calculate covariance matrix of centered matrix
    V = numpy.cov(C.T)
    
    # eigen decomposition of covariance matrix
    values, vectors = numpy.linalg.eigh(V)
    index = numpy.argsort(values)[::-1][:n]
    vectors = vectors[:, index]
    
    # project data
    X_pc = vectors.T.dot(C.T).T
    return X_pc, index

def modified_union(X, DF, percent):
    N, V = X.shape

    # Document Frequency
    df = DF.sum(0) 
    df /= max(df)

    # Term Variance
    tv = numpy.zeros(V, dtype=float)
    for i in range(V):
        xbar = numpy.average(X[:, i])
        for j in range(N):
            xi = X[j, i]
            tv[i] += (xi - xbar) ** 2
        tv[i] /= N
    tv /= max(tv)

    # Modified Unioun
    n = (percent * V) // 100
    dftv = df + tv
    index = numpy.argsort(dftv)[::-1][:n]
    return X[:, index], index
