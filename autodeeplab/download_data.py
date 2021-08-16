import os
import boto3
import sys

def download_data_from_s3(task):
    '''Download pde data from s3 to store in temp directory'''

    s3_bucket = "pde-xd"
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

    s3 = boto3.client("s3")
    os.makedirs(download_directory, exist_ok=True)

    for data_file in data_files:
        filepath = os.path.join(download_directory, data_file)
        s3_loc = os.path.join(s3_path, data_file) if s3_path is not None else data_file
        if not os.path.exists(filepath):
            s3.download_file(s3_bucket, s3_loc, filepath)

    return None

def main():
    task = sys.argv[1]
    download_data_from_s3(task)

if __name__ == '__main__':
    main()
