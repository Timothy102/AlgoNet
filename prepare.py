import numpy as np
import time
from tqdm import tqdm

from sorting import algo_list, merge_sort, bubble_sort, insertion_sort, quicksort, bucket_sort, countInv

TRAINING_SIZE = 100000
CLASSES = len(algo_list)

data = np.zeros((TRAINING_SIZE,2))
labels = np.zeros((TRAINING_SIZE,5))

### THE PLAN ###
# random array generation, extract info from array that might be helpful for each sort, also length and številka zamenjav potrebnih, check the book about dp


## from 1 to 1000 of size 1 to 1000 
def random_array_generator(upto = 1000):
    k = np.random.randint(1,high=upto, size = 1)
    return list(np.random.randint(1,size=int(k),high=upto, dtype=int))
    ## tle dodaš pol extract data


## tle dubi vse informacije o arrayu. 
def extract_data(arr):
    return [len(arr), countInv(arr)]

def time_exe(func,arr):
    start = time.time()
    func(arr)
    end = time.time()
    return end - start

def np_to_csv(arr, file = 'data.csv'):
    np.savetxt(file, arr, delimiter=",")
    
def prepare_data():
    for i in tqdm(range(TRAINING_SIZE)):
        arr = random_array_generator()
        min_time, min_index = 1e5,0
        for j in range(len(algo_list)):
            t = time_exe(algo_list[j], arr)
            if t > min_time : 
                min_time = t
                min_index = j
    labels[i][min_index] = 1
    data[i] = extract_data(arr)
    return data, labels

def onehotencode(x):
    y = np.zeros((CLASSES,1))
    y[x] = 1
    return y

data, targets = prepare_data()
print(data.shape,targets.shape)
