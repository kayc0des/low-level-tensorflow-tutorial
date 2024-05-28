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
    


y = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
one_hot_y = one_hot_encode(y, 10)
print(one_hot_y)


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