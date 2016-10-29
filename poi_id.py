#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from tester import test_classifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

#features_list contains the features that will be used in the pipeline.
features_list = ['poi','salary', 'bonus', 'total_stock_value','exercised_stock_options',
                 'bonus_salary_ratio','total_payments_and_stock']

'''
#All my features:
'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi','bonus_salary_ratio',
                 'total_payments_and_stock','to_and_from_poi_emails','total_poi_emails','percent_of_poi_to_emails',
                 'percent_of_poi_from_emails','percent_poi_emails'

'salary', 'bonus', 'total_stock_value','exercised_stock_options',
                 'bonus_salary_ratio','total_payments_and_stock'
'''

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Find the total number of data points.
data_points = 0.
for name in data_dict:
    for feature in data_dict[name]:
        data_points += 1
print "total data points: ", data_points

#Find the number of POI and non-POI data points.
POI_data_points = 0.
for name in data_dict:
    if data_dict[name]['poi'] == 1:
        for feature in data_dict[name]:
            POI_data_points += 1
print "POI dp: ", POI_data_points

non_POI_data_points = data_points - POI_data_points
print "non-POP dp: ", non_POI_data_points

percent_POI_data_points = POI_data_points / data_points
print "POI% dp: ", percent_POI_data_points

percent_non_POI_data_points = 1 - percent_POI_data_points
print "non-POI% dp: ", percent_non_POI_data_points

#Find total number of employees and POIs.
total_names = 0.
for name in data_dict:
    total_names += 1
print "total names: ", total_names

poi = 0
for name in data_dict:
    if data_dict[name]['poi'] == 1:
        poi += 1
print "number of POIs: ", poi

#Find how many missing data points are in each feature for POIs and non-POIs.
POI_nan = {}
non_POI_nan = {}

for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name]['poi'] == 1:
            if data_dict[name][feature] == 'NaN':
                if feature not in POI_nan:
                    POI_nan[feature] = 0
                POI_nan[feature] += 1
        if data_dict[name]['poi'] == 0:
            if data_dict[name][feature] == 'NaN':
                if feature not in non_POI_nan:
                    non_POI_nan[feature] = 0
                non_POI_nan[feature] += 1

print "Number of POI NaNs: ", POI_nan
print "Number of non-POI NaNs: ", non_POI_nan

#Search through data_dict to find any unexpected names.
'''
for name in data_dict:
    print name
'''
#'TOTAL', and 'THE TRAVEL AGENCY IN THE PARK' are not employees who worked at Enron, so they will be removed.
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

#If an employee at Enron [name] has a feature with a value of 'NaN', add 1 to their name in nan_count.
nan_count = {}
for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name][feature] != 'NaN':
            if name not in nan_count:
                nan_count[name] = 0
            nan_count[name] += 1

#print nan_count

#print the names of any employees with a value of 1. This means that all features except for 'poi' are NaN.
'''
for name, value in nan_count.iteritems():
    if value == 1:
        print name, value
'''

#'LOCKHART EUGENE E' has a value of 1, therefore there is very little to be learned from him and he will be removed
#from the data set.

data_dict.pop("LOCKHART EUGENE E")

#Validate the values of total_payments in the data. 
for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature] = 0
    if data_dict[name]['total_payments'] != (data_dict[name]['salary'] + data_dict[name]['bonus'] + data_dict[name]['long_term_incentive'] +
                                             data_dict[name]['deferred_income'] + data_dict[name]['deferral_payments'] + data_dict[name]['loan_advances'] +
                                             data_dict[name]['other'] + data_dict[name]['expenses'] + data_dict[name]['director_fees']):
        print "incorrect total_payments: ", name

data_dict['BELFER ROBERT']['total_payments'] = (data_dict['BELFER ROBERT']['salary'] + data_dict['BELFER ROBERT']['bonus'] + data_dict['BELFER ROBERT']['long_term_incentive'] +
                                                data_dict['BELFER ROBERT']['deferred_income'] + data_dict['BELFER ROBERT']['deferral_payments'] + data_dict['BELFER ROBERT']['loan_advances'] +
                                                data_dict['BELFER ROBERT']['other'] + data_dict['BELFER ROBERT']['expenses'] + data_dict['BELFER ROBERT']['director_fees'])

data_dict['BHATNAGAR SANJAY']['total_payments'] = (data_dict['BHATNAGAR SANJAY']['salary'] + data_dict['BHATNAGAR SANJAY']['bonus'] + data_dict['BHATNAGAR SANJAY']['long_term_incentive'] +
                                                   data_dict['BHATNAGAR SANJAY']['deferred_income'] + data_dict['BHATNAGAR SANJAY']['deferral_payments'] + data_dict['BHATNAGAR SANJAY']['loan_advances'] +
                                                   data_dict['BHATNAGAR SANJAY']['other'] + data_dict['BHATNAGAR SANJAY']['expenses'] + data_dict['BHATNAGAR SANJAY']['director_fees'])

#Validate the values of total_stock_value in the data.
for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature] = 0
    if data_dict[name]['total_stock_value'] != (data_dict[name]['exercised_stock_options'] + data_dict[name]['restricted_stock'] +
                                             data_dict[name]['restricted_stock_deferred']):
        print "incorrect total_stock_value: ", name

data_dict['BELFER ROBERT']['total_stock_value'] != (data_dict['BELFER ROBERT']['exercised_stock_options'] + data_dict['BELFER ROBERT']['restricted_stock'] +
                                             data_dict['BELFER ROBERT']['restricted_stock_deferred'])

data_dict['BHATNAGAR SANJAY']['total_stock_value'] != (data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] + data_dict['BHATNAGAR SANJAY']['restricted_stock'] +
                                             data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'])
    

#A new feature is created by dividing the first feature, by the second feature. NaN values are set to 0 for the
#first feature, and if the second feature has a value of NaN, then the new feature's value is set to 0, because
#we cannot divide by 0.
def new_feature_2_inputs_divide(new, first, second):
    for name in data_dict:
        if data_dict[name][first] == 'NaN':
            data_dict[name][first] = 0
        if data_dict[name][second] == 'NaN' or data_dict[name][second] == 0:
            data_dict[name][new] = 0
        else:
            data_dict[name][new] = (float(data_dict[name][first]) / float(data_dict[name][second]))

#A new feature is created by adding the first feature with the second. NaN values are set to 0.
def new_feature_2_inputs_add(new, first, second):
    for name in data_dict:
        if data_dict[name][first] == 'NaN':
            data_dict[name][first] = 0
        if data_dict[name][second] == 'NaN':
            data_dict[name][second] = 0
        data_dict[name][new] = (int(data_dict[name][first]) + int(data_dict[name][second]))

#A new feature is created by adding the first & second features together (numerator), as well as the third & fourth
#(denominator), then dividing the numerator by the denominator. All features have their NaN values set to 0, but if
#the third and fourth features have NaN values, the new feature has its value set to 0, because you cannot divide by 0.        
def new_feature_4_inputs_divide(new,first,second,third,fourth):
    for name in data_dict:
        if data_dict[name][first] == 'NaN':
            data_dict[name][first] = 0
        if data_dict[name][second] == 'NaN':
            data_dict[name][second] = 0
        if data_dict[name][third] == 'NaN':
            data_dict[name][third] = 0
        if data_dict[name][fourth] == 'NaN':
            data_dict[name][fourth] = 0
        if data_dict[name][third] == 'NaN' and data_dict[name][fourth] == 'NaN':
            data_dict[name][new] = 0.
        if data_dict[name][third] == 0 and data_dict[name][fourth] == 0:
            data_dict[name][new] = 0.
        else:
            data_dict[name][new] = (float(int(data_dict[name][first]) +
                                          int(data_dict[name][second])) /
                                    float(int(data_dict[name][third]) +
                                          int(data_dict[name][fourth])))            
            
#Create the new features
new_feature_2_inputs_divide('bonus_salary_ratio', 'bonus', 'salary')
new_feature_2_inputs_add('total_payments_and_stock', 'total_payments', 'total_stock_value')
new_feature_2_inputs_add('to_and_from_poi_emails', 'from_poi_to_this_person','from_this_person_to_poi')
new_feature_2_inputs_add('total_poi_emails', 'to_and_from_poi_emails','shared_receipt_with_poi')
new_feature_2_inputs_divide('percent_of_poi_to_emails', 'from_this_person_to_poi','to_messages')
new_feature_2_inputs_divide('percent_of_poi_from_emails', 'from_poi_to_this_person','from_messages')
new_feature_4_inputs_divide('percent_poi_emails','from_poi_to_this_person','from_this_person_to_poi',
                            'to_messages','from_messages')
    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Draw a plot comparing two features: f1_name and f2_name, along with their prediction line: pred.
def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):

    #plot each cluster with a different color--add more colors for
    #drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    #place red stars over points that are POIs (to see them better)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

feature_1 = "to_and_from_poi_emails"
feature_2 = "total_poi_emails"
poi  = "poi"
plot_features_list = [poi, feature_1, feature_2]
plot_data = featureFormat(data_dict, plot_features_list)
poi, plot_features = targetFeatureSplit( plot_data )
pred = KMeans(n_clusters=2).fit_predict(plot_features)

#Draw(pred, plot_features, poi, mark_poi=True, name="clusters1.pdf", f1_name=feature_1, f2_name=feature_2)

#Prepare data fro cross_validation
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

sss = StratifiedShuffleSplit(labels, 200, test_size=0.1, random_state=42)

pipe = Pipeline(steps=[('min_max_scaler', MinMaxScaler()),
                       #('select_KBest', SelectKBest()),
                       ('pca', PCA()),
                       #('kmeans', KMeans()),
                       ('logistic_regression', LogisticRegression()),
                       #('tree', DecisionTreeClassifier()),
                       #('svc', SVC()),
                       #('knc', KNeighborsClassifier()),
                       #('ada', AdaBoostClassifier())
                      ])

parameters = dict(#select_KBest__k= [3],
                  pca__n_components= [5],
                  pca__whiten = [False],
                  #kmeans__n_clusters = [3,4],
                  #kmeans__n_init = [2,5,8],
                  logistic_regression__C= [1],
                  logistic_regression__class_weight = [{True:2, False:1}],
                  logistic_regression__fit_intercept = [False],
                  #logistic_regression__dual = [True],
                  #logistic_regression__penalty = ['l2'],
                  #tree__max_features = [1],
                  #tree__min_samples_split = [10],
                  #tree__criterion = ['gini'],
                  #svc__gamma = [0.05,0.1,5],
                  #svc__C = [10,20],
                  #svc__kernel = ['rbf'],
                  #knc__n_neighbors = [2,3,5],
                  #knc__algorithm = ['auto'],
                  #knc__leaf_size = [1,2,3],
                  #ada__n_estimators = [90],
                  #ada__learning_rate = [5],
                  #ada__algorithm = ['SAMME.R']
          )

#Execute pipeline via GridSearchCV
grid_search = GridSearchCV(pipe, parameters, n_jobs = 1, cv = sss, scoring='f1', verbose = 2)

grid_search.fit(features, labels)

print ''
print grid_search.best_estimator_

#print the best POI identifier score and the best parameters.
print ''
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
       print '\t%s: %r' % (param_name, best_parameters[param_name])
'''
#print the best features and their scores.
features_k= grid_search.best_params_['select_KBest__k']
skb_k=SelectKBest(f_classif, k=features_k)
skb_k.fit_transform(features_train, labels_train) 
feature_scores = skb_k.scores_
features_selected=[features_list[i+1]for i in skb_k.get_support(indices=True)]
features_scores_selected=[feature_scores[i]for i in skb_k.get_support(indices=True)]
print ' '
print 'Selected Features', features_selected
print 'Feature Scores', features_scores_selected
'''

#set clf equal to the best POI identifier.
clf = grid_search.best_estimator_

#Dump classifier, dataset, and features_list so anyone can check results. 

dump_classifier_and_data(clf, my_dataset, features_list)

#Below are the highest scores for each algorithm I used, and their respective parameters.
#KNeighborsClassifier is not listed below because its best score was never competitively high.
'''
LogisticRegression
Best score: 0.448
Best parameters set:
	logistic_regression__C: 1
	logistic_regression__class_weight: {False: 1, True: 9}
	logistic_regression__fit_intercept: False
	pca__n_components: 3
	pca__whiten: False
	select_KBest__k: 8

DecisionTreeClassifier
Best score: 0.399
Best parameters set:
	pca__n_components: 3
	pca__whiten: False
	select_KBest__k: 4
	tree__criterion: 'gini'
	tree__max_features: 1
	tree__min_samples_split: 10

SVC
Best score: 0.335
Best parameters set:
	pca__n_components: 1
	pca__whiten: True
	select_KBest__k: 4
	svc__C: 0.1
	svc__gamma: 0.0001
	svc__max_iter: 3

AdaBoostClassifier
Best score: 0.331
Best parameters set:
	ada__algorithm: 'SAMME.R'
	ada__learning_rate: 5
	ada__n_estimators: 90
	pca__n_components: 1
	pca__whiten: True
	select_KBest__k: 3
'''

