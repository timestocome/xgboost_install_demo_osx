#!/python   

import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb

################################################################
# meh getting about 65% accuracy at best
##################################################################




# load saved txt file that was created in PrepDataForXGBoost into buffer
# files are in libsvm format
train = xgb.DMatrix('train.svm')
test = xgb.DMatrix('test.svm')




# specify parameters via map, current settings are the default
# 'booster':gbtree
# 'silent':0
# 'eta':0.3             # step size to prevent overfitting 0.0-1.0
# 'max_depth':6         # 1, inf 
param = {}
param['objective'] = 'multi:softmax'
param['max_depth'] = 32
param['silent'] = 1
param['num_class'] = 39
param['eta'] = 0.4
param['gamma'] = 1.0

# specify validations set to watch performance
watchlist = [(test, 'eval'), (train, 'train')]
num_round = 100                                     # number of times to loop through data ?


# train on training data
bst = xgb.train(param, train, num_round, watchlist)


# this is prediction on test data
preds = bst.predict(test)
labels = test.get_label()
preds *= 10
labels *= 10
correct = sum( preds == labels ) / float(len(preds))
print('*************  Correct predictions on test set: ', correct)



#############################################################################################
#???? predictions on kaggle data
#############################################################################################
kaggle = xgb.DMatrix('kaggle.svm')
kaggle_predictions = bst.predict(kaggle)
print("############  kaggle predictions ########################")
print(kaggle_predictions)
np.savetxt("kaggle_predictions.txt", kaggle)

"""
# need to output this to appropriately formatted text file for submitting




submission = pd.DataFrame(p, columns=['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT', 'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS'])

    
submission['Id'] = validation_data['Id']
print(submission)




submission.to_csv("mySubmission.csv", index=True )



#########################################################################
# save model to file
#########################################################################
# write model file to dist
bst.save_model('0001.model')
bst.dump_model('dump.raw.txt')



# save dmatrix into binary buffer
dtest.save_binary('dtest.buffer')

# save model
bst.save_model('xgb.model')

# load model and data in
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('test.buffer')
preds2 = bst2.predict(dtest2)

# assert they are the same
assert np.sum(np.abs(preds2-preds)) == 0



# alternatively, you can pickle the booster
pks = pickle.dumps(bst2)

# load model and data in
bst3 = pickle.loads(pks)
preds3 = bst3.predict(dtest2)

# assert they are the same
assert np.sum(np.abs(preds3-preds)) == 0

"""