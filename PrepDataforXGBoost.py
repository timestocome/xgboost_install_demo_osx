#!/python


import numpy as np
import pandas as pd

from datetime import datetime
from sklearn import cross_validation
from sklearn.datasets import dump_svmlight_file


import LoadData




# change categories to numbers,
# scale everything for use in NN
# parse date into min, hour, month, day, year
def prepData():
    
    # load up files from disk
    training_data, kaggle_data = LoadData.load_data()    
    features_in = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address' 'X', 'Y']
    

    # break dates into month, day, year, day of week, hour 
    # categorize category, month, day, year, dow, hour, district
    # scale lat (y), long(x)
    training_data['Year'] = (pd.DatetimeIndex(training_data['Dates']).year) 
    training_data['Month'] = (pd.DatetimeIndex(training_data['Dates']).month)
    training_data['Day'] = (pd.DatetimeIndex(training_data['Dates']).day)
    training_data['Hour'] = (pd.DatetimeIndex(training_data['Dates']).hour)
    training_data['Minute'] = (pd.DatetimeIndex(training_data['Dates']).minute)

    kaggle_data['Year'] = (pd.DatetimeIndex(kaggle_data['Dates']).year) 
    kaggle_data['Month'] = (pd.DatetimeIndex(kaggle_data['Dates']).month)
    kaggle_data['Day'] = (pd.DatetimeIndex(kaggle_data['Dates']).day)
    kaggle_data['Hour'] = (pd.DatetimeIndex(kaggle_data['Dates']).hour)
    kaggle_data['Minute'] = (pd.DatetimeIndex(kaggle_data['Dates']).minute)
    


    # cast date as unix time
    training_data['UnixTime'] = (pd.DatetimeIndex(training_data['Dates'])).astype(np.int64) / 10000000000
    kaggle_data['UnixTime'] = (pd.DatetimeIndex(kaggle_data['Dates'])).astype(np.int64) / 10000000000

   
    # day of week to number
    sorted_days = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')
    def dayOfWeekNumber(d):
        return sorted_days.index(d)
    training_data['DayNumber'] = (training_data['DayOfWeek'].apply(dayOfWeekNumber))
    kaggle_data['DayNumber'] = (kaggle_data['DayOfWeek'].apply(dayOfWeekNumber))
    
    
    # set up an id number for each category from alphabetical list
    # add to training_data
    categories = pd.unique(training_data['Category'])
    sorted_categories = (np.sort(categories)).tolist()

    def categoryNumber(category):
        return sorted_categories.index(category)
    training_data['CategoryNumber'] = training_data['Category'].apply(categoryNumber)
    
    # no categories for validation data, that's what we're trying to figure out
    # add output array for validation set just for convience 
    kaggle_data['CategoryNumber'] = 0
  
    print("min/max category", min(training_data['CategoryNumber']), max(training_data['CategoryNumber']))
    
    
    
    districts = pd.unique(training_data['PdDistrict'])
    sorted_districts = (np.sort(districts)).tolist()
    
    def districtNumber(district):
        return sorted_districts.index(district)
    training_data['DistrictNumber'] = (training_data['PdDistrict'].apply(districtNumber))
    kaggle_data['DistrictNumber'] = (kaggle_data['PdDistrict'].apply(districtNumber))
    
    
    
  

    # split inputs from outputs
    features = ['Year', 'Month', 'Day', 'Hour', 'X', 'Y', 'DayNumber', 'DistrictNumber', 'CategoryNumber']
    
    training_data = training_data[features]
    print("pre split ", len(training_data))
    
      # split training and testing
      ##### to do , training and testing might contain some duplicates? how to avoid this?
    testing_data = training_data.sample(frac=0.2, replace=False)
    training_data = training_data.sample(frac=0.8, replace=False)
    
    print("post split", len(training_data))
    print("test", len(testing_data))
    
    
    
    data = np.array(training_data)
    x = data[:, 0:8]
    y = data[:, 8]
    dump_svmlight_file(x, y, 'train.svm')
    
    
    data = np.array(testing_data)
    x = data[:, 0:8]
    y = data[:, 8]
    dump_svmlight_file(x, y, 'test.svm')
    
    
    kaggle_data = kaggle_data[features]
    data = np.array(kaggle_data)
    x = data[:, 0:8]
    y = data[:, 8]
    dump_svmlight_file(x, y, 'kaggle.svm')
    
    # sanity check data
    print(training_data.head())
    print(x[0])
    print(y[0])
    
    
    
    
prepData()