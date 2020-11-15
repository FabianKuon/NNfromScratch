import pickle
import requests
import os.path

def FileWatch():
    if os.path.exists('cifar-10-batches-py'):
        print('cifar data set available in current folder')
    else:
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        r = requests.get(url, allow_redirects=True)
        open('cifar-10-python.tar.gz', 'wb').write(r.content)
        print('Downloading cifar data set...')

def main():
    FileWatch()

if __name__=='main':
    main()

