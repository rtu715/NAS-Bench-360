import os
from urllib.request import urlretrieve
import sys

def download_data_from_s3(task):
    '''Download pde data from s3 to store in temp directory'''
    s3_base = "https://pde-xd.s3.amazonaws.com"
    download_directory = "."
    
    if task == 'ECG':
        #urllib.urlretrieve("https://pde-xd.s3.amazonaws.com/ECG/challenge2017.pkl", filename="challenge2017.pkl")
        data_files = ['challenge2017.pkl']
        s3_folder = 'ECG'

    elif task == 'satellite':
        data_files = ["satellite_train.npy", "satellite_test.npy"]
        s3_folder = 'satellite'
    
    elif task == 'deepsea':
        data_files = ["deepsea_filtered.npz"]
        s3_folder = 'deepsea'

    else:
        raise NotImplementedError

    os.makedirs(download_directory, exist_ok=True)

    for data_file in data_files:
        if not os.path.exists(data_file):
            fileurl = s3_base + '/' + s3_folder + '/' + data_file
            urlretrieve(fileurl)

    return None

def main():
    task = sys.argv[1]
    download_data_from_s3(task)

if __name__ == '__main__':
    main()
