"""Weighted model based on random forest classification"""

# Author: Zhaoen Su <suzhaoen@gmail.com>

# Not converted into a class yet.
# However, with the same data processing function in the k-NN classification file
# the functions below implement my modified random forest method
# Read the report for more details

def user_predict(rfclassifier, Xtrain, Ytrain, xtest, weight, bin_horizontal, bin_vertical):
    """predict the bin of the dropoff location for a single user
    Parameters
    ===========
    Xtrain : pd dataframe, shape (n_test_samples, n_features)
        The training feature data
    Ytrain : pd dataframe, shape (n_test_samples, n_targets+1)
        The training target data
    Xtest : pd dataframe, shape (n_test_samples, n_features)
        The test features data
    weight : float
        The weight parameter of the model
    lat_min : float
        The minium latitude of the bound
    lng_min : float
        The minium longitude of the bound
    width : float
        The latitude range of the bound
    length : float
        The longtitude range of the bound
    bin_horizontal : int
        The number of bins in horizontal direction
    bin_vertical : int
        The number of bins in vertical direction
    Returns
    ========
    pred : pd dataframe
        The predictive bin of the user
        The dataframe has a single column: pred_coor_val
        pred_coor_val is the number of the preditive bin
    probabilities : list of floats
        The probabilities distribtution of the discretized bins
        It is ready to be returned.
    """
    #initialize the probabilities for dropoff at each bin as uniformly
    proba_outcomes_user = np.array([1/(bin_horizontal*bin_vertical)]*(int(bin_horizontal*bin_vertical)))

    if Xtrain.loc[Xtrain.uid == xtest.uid].shape[0] > 0:
        Xtrain_user = Xtrain.loc[Xtrain.uid == xtest.uid, ['begintrip_lat', 'begintrip_lng', 'is_weekend', 'time']]
        Ytrain_user = Ytrain.loc[Ytrain.uid == xtest.uid, ['coor_val']]
        rfc_user = rfclassifier.fit(Xtrain_user, Ytrain_user.coor_val.ravel())
        xtest_user = xtest.drop('uid', axis=0)
        xtest_user = xtest_user.reshape(1, -1)
        possible_outcomes_user = np.unique(np.array(Ytrain_user.coor_val))
        probabilities_user = rfc_user.predict_proba(xtest_user)
    
        for i in range(len(possible_outcomes_user)):
            proba_outcomes_user[int(possible_outcomes_user[i])] += probabilities_user[0][i]

        proba_outcomes_user = proba_outcomes_user**(weight)
        
    # sampling to speed up
    itrain, itest = train_test_split(range(Ytrain.shape[0]), train_size=0.2)
    mask=np.ones(Ytrain.shape[0], dtype='int')
    mask[itrain]=1
    mask[itest]=0
    mask = (mask==1)
    Xtrain, Ytrain = Xtrain[mask], Ytrain[mask]
    
    Xtrain_all = Xtrain.loc[:, ['begintrip_lat', 'begintrip_lng', 'is_weekend', 'time']]
    Ytrain_all = Ytrain.loc[:, ['coor_val']]
    rfc_all = rfclassifier.fit(Xtrain_all, Ytrain_all.coor_val.ravel())
    xtest_all = xtest.drop('uid', axis=0)
    xtest_all = xtest_all.reshape(1, -1)
    possible_outcomes_all = np.unique(np.array(Ytrain_all.coor_val))
    probabilities_all = rfc_all.predict_proba(xtest_all)
    
    #initialize the probabilities for dropoff at each bin as uniformly
    proba_outcomes_all = np.array([1/(bin_horizontal*bin_vertical)]*(int(bin_horizontal*bin_vertical)))
    
    for i in range(len(possible_outcomes_all)):
        proba_outcomes_all[int(possible_outcomes_all[i])] += probabilities_all[0][i]
    
    probabilities = np.multiply(proba_outcomes_user,proba_outcomes_all)
    most_likely_coor_val = np.argmax(probabilities)
    
    normalized the probablilities distribution
    probabilities /= sum(probabilities)
    return pd.Series({'pred_coor_val': most_likely_coor_val})

def rf_predict(Xtrain, Ytrain, Xtest, weight, lat_min, lng_min, width, length, bin_horizontal, bin_vertical):
    """predict the dropoff location for test samples
        Implement the modified rf method
    Parameters
    ===========
    Xtrain : pd dataframe, shape (n_test_samples, n_features)
        The training feature data
    Ytrain : pd dataframe, shape (n_test_samples, n_targets+1)
        The training target data
    Xtest : pd dataframe, shape (n_test_samples, n_features)
        The test features data
    weight : float
        The weight parameter of the model
    lat_min : float
        The minium latitude of the bound
    lng_min : float
        The minium longitude of the bound
    width : float
        The latitude range of the bound
    length : float
        The longtitude range of the bound
    bin_horizontal : int
        The number of bins in horizontal direction
    bin_vertical : int
        The number of bins in vertical direction
    Returns
    ========
    pred : pd dataframe
        The prediction of the Xtest
        The dataframe includes pred_lat, pred_lng, pred_coor_val columns
        pred_coor_val is the number of the preditive bin
    """
    rfclassifier = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    pred = Xtest.apply(lambda row: user_predict(rfclassifier, Xtrain, Ytrain, row, weight, bin_horizontal, bin_vertical), axis = 1)
    pred = pred.join(pred.apply(lambda row: val2coordinates(row.pred_coor_val, lat_min, lng_min, width, length, bin_horizontal, bin_vertical), axis = 1))
    return pred
