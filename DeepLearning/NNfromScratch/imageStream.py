import pickle
import requests
import os.path
import os

def FileWatch():
    if os.path.exists('cifar-10-batches-py'):
        print('cifar data set available in current folder')
    else:
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        r = requests.get(url, allow_redirects=True)
        open('cifar-10-python.tar.gz', 'wb').write(r.content)
        print('Cifar data set downloaded successfully...')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    FileWatch()
    dict = unpickle('cifar-10-batches-py/data_batch_1')

if __name__=="__main__":
    main()

