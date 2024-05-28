## Classification Intranet - Project 14

import numpy as np

def one_hot_encode(Y, classes):
    '''
    Converts numeric label vector into a one hot matrix
    
    Param:
    Y -> np array with shape (m,)
    m -> number of examples
    classes -> maximum number of classes found in Y
    
    Returns:
    A one hot encoding of Y with shape (classes, m) or None on failure
    '''
    
    m = Y.shape[0] 
    
    # init one hot matrix
    one_hot = np.zeros((classes, m))
    
    for i in range(len(Y)):
        target = Y[i]
        one_hot[target, i] = 1
        
    return one_hot
    
def one_hot_decoding(one_hot):
    '''
    Converts a one hot matrix into a vector of labels
    
    Param:
    one_hot -> one hot matrix
    
    Returns:
    A np.array containing numeric labels
    ''' 
    return np.argmax(one_hot, axis=0)
    

y = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
one_hot_y = one_hot_encode(y, 10)
print(one_hot_y)
y_decoded = one_hot_decoding(one_hot_y)
print(y_decoded)


# Using sklearn
'''
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit([1, 2, 2, 6])

print(le.fit([1, 2, 2, 6]))
print(le.classes_)
print(le.transform([1, 2, 2, 6]))
print(le.inverse_transform([0, 0, 1, 2]))
'''