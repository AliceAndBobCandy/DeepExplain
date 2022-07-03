
import enum
import numpy as np

def one_hot_encoding(labels,max_label=1):
    '''
    change the labels to one hot expression
    labels: numpy array, (len,)
    max_label: max label of labels
    '''
    res = np.zeros((len(labels),max_label+1))
    for i,label in enumerate(labels):
        res[i,1] = 1
    return res