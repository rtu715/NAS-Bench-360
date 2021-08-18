import os
from urllib.request import urlretrieve
import sys

def download_data_from_s3(task):
    '''Download pde data from s3 to store in temp directory'''

    s3_base = "https://pde-xd.s3.amazonaws.com"
    download_directory = "."
    
    if task == 'darcyflow':
        data_files = ["piececonst_r421_N1024_smooth1.mat", "piececonst_r421_N1024_smooth2.mat"]
        s3_path = None

    elif task == 'protein':
        data_files = ['protein.zip']
        s3_path = None

    elif task == 'cosmic':
        data_files = ['deepCR.ACS-WFC.train.tar', 'deepCR.ACS-WFC.test.tar']
        s3_path = 'cosmic'

    else:
        raise NotImplementedError

    #os.makedirs(download_directory, exist_ok=True)

    for data_file in data_files:
        if not os.path.exists(data_file):
            if s3_path is not None:
                fileurl = s3_base + '/' + s3_path + '/' + data_file
            else: 
                fileurl = s3_base + '/' + data_file
            urlretrieve(fileurl, data_file)

    return None

def main():
    task = sys.argv[1]
    download_data_from_s3(task)

if __name__ == '__main__':
    main()
