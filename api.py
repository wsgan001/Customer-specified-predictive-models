#!/bin/python3
"""Weighted model based on k-nearest neighbors classification"""

# Author: Zhaoen Su <suzhaoen@gmail.com>

import numpy as np
import pandas as pd
import math
import time
import datetime
from sklearn.neighbors import KDTree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


def check_data_names(rides):
    """Check if the dataframe has the right columns
    Parameters
    ===========
    rides : pd dataframe
        The data
    Returns
    ========
    Raise ValueError if the columns are 
                     'begintrip_at', 
                     'begintrip_lat',
                     'begintrip_lng',
                     'dropoff_lat',
                     'dropoff_lng',
                     'uid'
    Return None given correct names
    """
    correct_names = ['begintrip_at',
                     'begintrip_lat',
                     'begintrip_lng',
                     'dropoff_lat',
                     'dropoff_lng',
                     'uid']
    names = rides.columns.tolist()
    if rides.shape[1] != len(correct_names) or correct_names.sort() != names.sort():
        raise ValueError("The columns should have and only have begintrip_at, begintrip_lat, begintrip_lng, dropoff_lat, dropoff_lng, uid, is_weekend, time.")
    
    
def check_X_names(Xtest):
    """Check if the dataframe of the feature data has the right columns
    Parameters
    ===========
    Xtest : pd dataframe
        The feature data
    Returns
    ========
    Raise ValueError if the columns are
                     'begintrip_at', 
                     'begintrip_lat',
                     'begintrip_lng',
                     'uid'
    Return None given correct names
    """
    correct_names = ['begintrip_lat',
                     'begintrip_lng',
                     'uid', 
                     'is_weekend', 
                     'time']
    names = Xtest.columns.tolist()
    if Xtest.shape[1] != len(correct_names) or correct_names.sort() != names.sort():
        raise ValueError("The columns should have and only have begintrip_lat, begintrip_lng, uid, is_weekend, time.")
   

def check_Y_names(Ytest):
    """Check if the dataframe of target data has the right columns
    Parameters
    ===========
    Ytest : pd dataframe
        The target data
    Returns
    ========
    Raise ValueError if the columns are
                     'dropoff_lat',
                     'dropoff_lng',
                     'uid'
    Return None given correct names
    """
    correct_names = ['dropoff_lat',
                     'dropoff_lng',
                     'uid']
    names = Ytest.columns.tolist()
    if Ytest.shape[1] != len(correct_names) or correct_names.sort() != names.sort():
        raise ValueError("The columns should have and only have dropoff_lat, dropoff_lng and uid.")
   

def geo_bound(rides):
    """Get the latitude and longtitude bounds of the samples
    Parameters
    ===========
    rides : pd dataframe
        The data
    Returns
    ========
    (lat_min, lat_max, lng_min, lng_max): tuple of floats
        The minimum latitude, the maximum latitude, 
        the minimum longitude, the maximum longitude in order.
        The four values are floats
    """
    lat_min = min(rides.begintrip_lat.min(), rides.dropoff_lat.min())
    lat_max = max(rides.begintrip_lat.max(), rides.dropoff_lat.max())
    lng_min = min(rides.begintrip_lng.min(), rides.dropoff_lng.min())
    lng_max = max(rides.begintrip_lng.max(), rides.dropoff_lng.max())
    return (lat_min, lat_max, lng_min, lng_max)

def set_bin_number(boundary, bin_number):
    """Get the numbers of discretized squared bins in the horizontal and vertical directions
    Parameters
    ===========
    boundary : tuple
        The boundary of the regime defined by (lat_min, lat_max, lng_min, lng_max)
    bin_number: int
        The number of discretized squared bins in the horizontal direction
    Returns : tuple of ints
    ========
        The numbers of discretized bins in the horizontal and vertical directions
    """
    lat_min, lat_max, lng_min, lng_max = boundary
    horizontal_vertical_ratio = math.cos((lat_max+lat_min)/2/180*(math.pi))
    bin_horizontal = bin_number
    bin_vertical = bin_number * ((lng_max - lng_min) // (((lat_max-lat_min)*horizontal_vertical_ratio)))
    return (bin_horizontal, bin_vertical)

def date_time_parser(date_time):
    """Parse a time-date stamp and get time and date quantitative features
    Parameters
    ===========
    data_time : string
        The time-data stamp. The 'begintrip_at' value.
        An example is 2015-02-28_20:27:09
    Returns
    ========
    is_weekend : int
        0 if the date is a weekend
        1 if the date is not a weekend
    time : float
        The time in hour. The minute and second values are converted into hour.  
    Return None if date_time has the wrong format.
    """
    date_time = date_time.split('_')
    if len(date_time) != 2: return
    
    date = [int(d) for d in date_time[0].split('-')]
    if len(date) != 3: return
    
    time = [int(t) for t in date_time[1].split(':')]
    if len(time) != 3: return
    
    date_object = datetime.date(date[0], date[1], date[2])
    # {1,2,...,7} represent {Monday,Tuesday ..., Sunday} in order
    day_of_week = date_object.isoweekday()
    day_of_week_cos = math.cos(day_of_week / 7 * 2 * math.pi)
    day_of_week_sin = math.sin(day_of_week / 7 * 2 * math.pi)
    is_weekend = 0
    if day_of_week in (6, 7):
        is_weekend = 1
    
    time = time[0] + time[1] / 60 + time[2] / 3600
    time_cos = math.cos(time / 24 * 2 * math.pi)
    time_sin = math.sin(time / 24 * 2 * math.pi)
    
    return (is_weekend, time)


def data_clearning(data_frame, boundary):
    """Check if the latitude and longitude values of the data in the dataframe of are within the boundary    
    Parameters
    ===========
    data_frame : pd dataframe
        The data
    boundary: tuples of floats
        The boundary defined by four lagitude and longtitude values: (lat_min, lat_max, lng_min, lng_max)
    Returns
    ========
    data_frame : pd dataframe
        The samples whose spatial features are outside the boundary is removed.
    """
    lat_min, lat_max, lng_min, lng_max = boundary
    return data_frame[(data_frame.is_weekend != None)\
                      & (data_frame.begintrip_lat >= lat_min)\
                      & (data_frame.begintrip_lat <= lat_max)\
                      & (data_frame.begintrip_lng >= lng_min)\
                      & (data_frame.begintrip_lng <= lng_max)\
                      & (data_frame.dropoff_lat >= lat_min)\
                      & (data_frame.dropoff_lat <= lat_max)\
                      & (data_frame.dropoff_lng >= lng_min)\
                      & (data_frame.dropoff_lng <= lng_max)]

def coordinates2val(lat_min, lng_min, width, length, x, y, bin_horizontal, bin_vertical):
    """Convert a (lat, lng) pair into a number
        The spatial regime is discretized into Parse a bin_horizontal * bin_vertical squared bins
        Starting from bottom-left corner, the val of the bin is 0;
        Go to the right, bin number is increased by 1 for each bin;
        After a row, go up to the next row and the val is increased by 1.
        Finally, the top-right corner bin has a value of bin_horizontal * bin_vertical - 1
    Parameters
    ===========
    lat_min : float
        The minium latitude of the bound
    lng_min : float
        The minium longitude of the bound
    width : float
        The latitude range of the bound
    length : float
        The longtitude range of the bound
    x : float
        The latitude of the sample
    y : float
        The longitude of the sample
    bin_horizontal : int
        The number of bins in horizontal direction
    bin_vertical : int
        The number of bins in vertical direction
    Returns
    ========
    val : int
        The val of the bin where the latitude and longitude of the sample locate
    """
    i = (x-lat_min) // (width / bin_horizontal)
    j = (y-lng_min) // (length / bin_vertical)

    if i == bin_horizontal: i -= 1
    if j == bin_vertical: j -= 1
    return j*bin_horizontal + i

def val2coordinates(val, lat_min, lng_min, width, length, bin_horizontal, bin_vertical):
    """Convert a the value of the bin back to the (lat, lng)
        The spatial regime is discretized into Parse a bin_horizontal * bin_vertical squared bins
        Starting from bottom-left corner, the val of the bin is 0;
        Go to the right, bin number is increased by 1 for each bin;
        After a row, go up to the next row and the val is increased by 1.
        Finally, the top-right corner bin has a value of bin_horizontal * bin_vertical - 1
    Parameters
    ===========
    val : int
        The val of the bin where the latitude and longitude of the sample locate
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
    (x, y) : tuple of floats
        x is the latitude of the sample
        y is the longitude of the sample
    """
    x = lat_min + (0.5 + val % bin_horizontal) * (width / bin_horizontal)
    y = lng_min + (0.5 + val // bin_horizontal) * (length / bin_vertical)
    return (x, y)

def float_convert(strs):
    return float(strs)
    

def data_preprocess(rides, bin_number):
    """Preprocess the data and return feature and target data sets.
        The following processes are performed:
        0 insure correct data type
        1 get the bound of the regime
        2 set descretization bin numbers in horizontal and vertical directions
        3 parse and add is_weekend and time features
        4 data clearning
        5 split DataFrame into feature and lable sets
        6 normalize the features, except the 'uid' column
    Parameters
    ===========
    rides : pd dataframe
        The data
    bin_number: int
        The number of bins in horizontal direction
    Returns
    ========
    X : pd dataframe
        The feature data
    Y : pd dataframe
        The target data
    """
    check_data_names(rides)
    rides['dropoff_lat'] = pd.to_numeric(rides['dropoff_lat'], errors='coerce')
    rides['dropoff_lng'] = pd.to_numeric(rides['dropoff_lng'], errors='coerce')
    rides['begintrip_lng'] = pd.to_numeric(rides['begintrip_lng'], errors='coerce')
    rides['begintrip_lng'] = pd.to_numeric(rides['begintrip_lng'], errors='coerce')
    boundary = geo_bound(rides)
    bin_numbers = set_bin_number(boundary, bin_number)   
    
    date_time = rides.begintrip_at.apply(date_time_parser).apply(pd.Series)
    date_time.columns = ['is_weekend','time']
    rides['is_weekend'] = date_time.is_weekend
    rides['time'] = date_time.time   
    
    rides = data_clearning(rides, boundary)     
    X = rides.drop(['begintrip_at', 'dropoff_lat', 'dropoff_lng'], axis=1)
    Y = rides.loc[:,['dropoff_lat', 'dropoff_lng', 'uid']]
    
    cols_to_norm = ['begintrip_lat', 'begintrip_lng', 'is_weekend', 'time']
    X[cols_to_norm] = X[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.std()))
    return (X, Y)
    
def split_data(X, Y, ratio):
    """Split the feature and target data into two parts
        The order of them is reserved.
    Parameters
    ===========
    X : pd dataframe
        The feature data
    Y : pd dataframe
        The target data
    Returns
    ========
    Xtrain : pd dataframe
        The feature data for training
    Ytrain : pd dataframe
        The target data for training
    Xtest : pd dataframe
        The feature data for validation
    Ytest : pd dataframe
        The target data for validation
    """
    itrain, itest = train_test_split(range(X.shape[0]), train_size=ratio)
    mask=np.ones(X.shape[0], dtype='int')
    mask[itrain]=1
    mask[itest]=0
    mask = (mask==1)

    Xtrain, Xtest, Ytrain, Ytest = X[mask], X[~mask], Y[mask], Y[~mask]
    n_samples = Xtrain.shape[0]
    n_features = Xtrain.shape[1]
    
    return (Xtrain, Ytrain, Xtest, Ytest)

class k_nearest_neighbors_modified():
    """Classifier implementing the k-nearest neighbors weighted vote.
    Read more in the report
    Parameters
    ----------
    boundary : tuple of list
        The latitude and longitude boundary, defined by (lat_min, lat_max, lng_min, lng_max)
    k_neighbors : int, optional (default = 100)
        The number of neighbors considered to vote
    weight_k_ratio : float, optional (default = 1)
        The weight parameter of the importance of the data of this user.
        Read more in the report.
    bin_number : int, optional (defulat = 10)
        The number of bins in horizontal direction
    """
    
    def __init__(self,boundary, k_neighbors = 100, weight_k_ratio = 1, bin_number = 10): 
        self.k_neighbors = k_neighbors
        self.weight_k_ratio = weight_k_ratio
        self.bin_number = bin_number
        self.lat_min, self.lat_max, self.lng_min, self.lng_max = boundary
        self.width = self.lat_max - self.lat_min
        self.length = self.lng_max - self.lng_min
        self.bin_horizontal, self.bin_vertical = set_bin_number(boundary, self.bin_number)
        self.y_pred = None
        
    def fit(self, Xtrain, Ytrain):
        """Fit the training data and add a 'coor_val' column to Ytrain
            'coor_val' is the value of the bin where the latitude and longitude of the sample locate
        Parameters
        ----------
        Xtrain : pd dataframe, shape (n_samples, n_features)
            The traning feature data
        Ytrain : pd dataframe, shape (n_samples, n_targets + 1)
            The traning target data. The 'uid' column is included.
        Returns
        -------
        None
        """
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Ytrain['coor_val'] = self.Ytrain.apply(lambda row: coordinates2val(self.lat_min, self.lng_min, self.width, self.length, row.dropoff_lat, row.dropoff_lng, self.bin_horizontal, self.bin_vertical), axis=1)
        self.Ytrain = self.Ytrain.loc[:, ['uid', 'coor_val']]
        
    def user_predict_coor(self, row):
        """Fit the training data and add a 'coor_val' column to Ytrain
            'coor_val' is the value of the bin where the latitude and longitude of the sample locate
        Parameters
        ----------
        Xtrain : pd dataframe, shape (n_samples, n_features)
            The traning feature data
        Ytrain : pd dataframe, shape (n_samples, n_targets + 1)
            The traning target data. The 'uid' column is included.
        Returns
        -------
        None
        """
        Xtrain = self.Xtrain.drop('uid', axis=1)
        xtest = row.drop('uid', axis=0)
        xtest = xtest.reshape(1, -1)
        tree = KDTree(Xtrain)
        ind = tree.query(xtest, k = self.k_neighbors, return_distance = False)
        # ind elements are the indices of the nearest neighbors
        cnt_ind = [1]*len(ind[0])# ind is a list of list
        vote_cnt = {}
        for i in range(len(ind[0])): 
            if self.Ytrain.loc[ind[0][i]]['uid'] == row['uid']:
                cnt_ind[i] += self.k_neighbors * self.weight_k_ratio
            label = self.Ytrain.loc[ind[0][i]]['coor_val']
            if label in vote_cnt:
                vote_cnt[label] += cnt_ind[i]
            else:
                vote_cnt[label] = cnt_ind[i]
        
        max_cnt = 0
        max_likely_val = 100
        for key in vote_cnt:
            if vote_cnt[key] > max_cnt:
                max_cnt = vote_cnt[key]
                max_likely_val = key

        lat, lng = val2coordinates(max_likely_val, self.lat_min, self.lng_min, self.width, self.length, self.bin_horizontal, self.bin_vertical)
        return pd.Series({'pred_lat': lat, 'pred_lng': lng, 'pred_coor_val':max_likely_val})
        
    def predict(self, Xtest):
        """Use the model to predict the dropped off latitude and longitude
        Parameters
        ----------
        Xtest : pd dataframe, shape (n_test_samples, n_features)
            The test feature data
        Returns
        -------
        y_pred : pd dataframe, shape (n_test_samples, n_targets)
            The latitude and longtitude of the center of the predictive dropped off bin
        """
        check_X_names(Xtest)
        self.Xtrain = self.Xtrain.reset_index(drop=True)
        self.Ytrain = self.Ytrain.reset_index(drop=True)
        self.y_pred = Xtest.apply(lambda row: self.user_predict_coor(row), axis = 1)
        return self.y_pred.loc[:, ['pred_lat', 'pred_lng']]
    
    def residue_measure(self, Ytest):
        """Get the residue between the true latitude and longitude and the predictive values
        Parameters
        ----------
        Ytest : pd dataframe, shape (n_test_samples, n_targets)
            The test target data
        Returns
        -------
        residue : float
            The residue between predictive and true locations
        """
        check_Y_names(Ytest)
        Ytest = Ytest.loc[:, ['dropoff_lat', 'dropoff_lng']]
        return np.sqrt(mean_squared_error(Ytest, self.y_pred.loc[:, ['pred_lat', 'pred_lng']]))
    
    def accuracy_measure(self, Ytest):
        """Get the accuracy that the true latitude and longitude locate in the predictive bin
        Parameters
        ----------
        Ytest : pd dataframe, shape (n_test_samples, n_targets)
            The test target data
        Returns
        ===========
        accuracy : float
            The accuracy of predicting the correct bins
        """
        check_Y_names(Ytest)
        Ytest['coor_val'] = Ytest.apply(lambda row: coordinates2val(self.lat_min, self.lng_min, self.width, self.length, row.dropoff_lat, row.dropoff_lng, self.bin_horizontal, self.bin_vertical), axis=1)
        return np.mean(Ytest.coor_val == self.y_pred.pred_coor_val)
 

if __name__ == "__main__":
    bin_number = 10
    k = 100
    weight = 10
    
    rides = pd.read_csv('data/hw1_train.csv')
    Xtrain, Ytrain = data_preprocess(rides, bin_number)
    boundary = geo_bound(rides)
    rides_test = pd.read_csv('data/hw1_test.csv')
    Xtest, Ytest = data_preprocess(rides_test, bin_number)
    
    # split data for validation, if necessary
    #Xtrain, Ytrain, Xtest, Ytest = split_data(X, Y, 0.4)
    
    knnm = k_nearest_neighbors_modified(boundary, k, weight, bin_number)
    knnm.fit(Xtrain, Ytrain)
    
    # use partial number for test
    test_num = 200
    Xtest = Xtest[0:test_num]
    Ytest = Ytest[0:test_num]
    yf = knnm.predict(Xtest)
    
    print(knnm.residue_measure(Ytest))
    print(knnm.accuracy_measure(Ytest))
