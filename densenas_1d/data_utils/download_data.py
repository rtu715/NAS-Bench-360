import os 
import boto3

def download_from_s3(s3_bucket, task, download_dir):
    s3 = boto3.client("s3")
    
    if task == 'ECG':
        data_files = ["challenge2017.pkl"]
        s3_folder = 'ECG'

    elif task == 'satellite':
        data_files = ["satellite_train.npy", "satellite_test.npy"]
        s3_folder = 'satellite'

    elif task == 'deepsea':
        data_files = ["deepsea_filtered.npz"]
        s3_folder = 'deepsea'

    else:
        raise NotImplementedError

    for data_file in data_files:
        filepath = os.path.join(download_dir, data_file)
        if s3_folder is not None:
            s3_path = os.path.join(s3_folder, data_file)
        else:
            s3_path = data_file
        if not os.path.exists(filepath):
            s3.download_file(s3_bucket, s3_path, filepath)    

    return

