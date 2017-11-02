import os
import subprocess


DIRPATH = './dataset'

if not os.path.exists(DIRPATH):
    os.makedirs(DIRPATH)

url_base = 'http://yann.lecun.com/exdb/mnist/'
file_names = ['train-images-idx3-ubyte.gz',
              'train-labels-idx1-ubyte.gz',
              't10k-images-idx3-ubyte.gz',
              't10k-labels-idx1-ubyte.gz']
for file_name in file_names:
    url = (url_base + file_name).format(**locals())
    out_path = os.path.join(DIRPATH, file_name)
    cmd = ['curl', url, '-o', out_path]
    print('Downloading ', file_name)
    subprocess.call(cmd)
    cmd = ['gzip', '-d', out_path]
    print('Decompressing ', file_name)
    subprocess.call(cmd)
