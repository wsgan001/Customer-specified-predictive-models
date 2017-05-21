# Customer-specified-predictive-models

Built models to predict an individual user’s multi-output choice (location). It integrated the few (~8 records for each user on average) but informative records of the given user and abundant but less relevant records of other users based on modified random forest and k-NN classifiers, and reached a accuracy that is 2 times higher than that by standard algorithms. 

Develop approaches to predict where a Uber or Lyft user would ask to be dropped off, given the user's ID, the time, date, and location of the pickup, based on the trip records of all users in a same city. 

The problem has two challenges. 
•	The predictive model is user specific. The trip records of the given user is the most informative. However, the average amount of trip records per user is just 8, which is way insufficient.  
•	The location to be predicted consists of two values, latitude and longitude. Thus the problem is a non-decomposable multi-output machine learning problem.  If the dependence between latitude and longitude is not reserved, the model would loss performance. 

We first preprocess the data by cleaning the data with Python libraries (pandas and numpy) and visualizing the data with matplot and seabone. We then explore various methods to engineer the features, time, date, and locations.  The final choices are made based on the performance when the full models are implemented and tested. It is found that, (1) a mapping to a 2D circle is helpful to engineer the time feature. (2) Using a binary value to indicate where a date is a weekday is sufficient. (3) the locations should be inside boundaries such that the model predicts well. After the feature engineering, I further process the data by zeroing the center, normalizing or stretching each features according to their importance.

•	To overcome the problem that each user has insufficient records, I make the utmost of the data by building weighted models that integrates the data of the given user that is less but informative and the data of all other users that is large but less relevant. 

•	To solve the multi-output machine learning problem while maintaining the important dependence between latitude and longitude of locations, I adapt K-nearest neighbors and random forest algorithms that naturally support multi-output problems. 
To fulfill the weighted model, I modify and develop classification and regression algorithms based on K-nearest neighbors and random forest. For example, I use the KD-tree data structure in the scikit-learning library for the k-NN algorithm such that the time complexity is reduced from O(N*N) to O(NlgN). 

To evaluate and compare the methods, two measures, accuracy and residue, are used. After hyperparameter optimization and testing, I find that accuracy is a more informative measure than residue. Closely related, the classification methods outperform the regression methods. Both the modified classification methods after optimization yield accuracies of 1/5 of predicting the true discretized bin among 300 bins of the whole regime. With visualization of the results, it is found that they are also 2 times better than those obtained with standard machine learning algorithms. Comprehensive process can be seen in the write-up.

