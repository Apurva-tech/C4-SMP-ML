import numpy as np
def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)  #load the file using numpy
    print('Dimensions: ',data.shape)
    print(data[1:6,:])                             #print first six samples of data
    return(data)
