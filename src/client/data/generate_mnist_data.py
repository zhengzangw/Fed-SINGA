from data import mnist
import numpy as np
from tqdm import tqdm

def split():
    
    train_x, train_y, val_x, val_y = mnist.load()


    for i in tqdm(range(10)):
        train = train_x[np.where(train_y==i)]
        val = val_x[np.where(val_y==i)]
        train = train.reshape(len(train), 28*28)
        val = val.reshape(len(val), 28*28)
        fn1 = "mnist_train_" + str(i) +".csv"
        fn2 = "mnist_val_" + str(i) +".csv"
        np.savetxt(fn1, train, delimiter=',')
        np.savetxt(fn2, val, delimiter=',')


if __name__=='__main__':
    split()
