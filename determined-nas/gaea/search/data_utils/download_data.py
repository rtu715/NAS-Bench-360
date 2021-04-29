import os 
import boto3

def download_from_s3(s3_bucket, task, download_dir):
    s3 = boto3.client("s3")
    
    if task == 'spherical':
        data_files = ["s2_mnist.gz"]
        s3_folder = None

    elif task == 'sEMG':
        #data_files = ["saved_evaluation_dataset_test0.npy", "saved_evaluation_dataset_test1.npy",
        #              "saved_evaluation_dataset_training.npy", "saved_pre_training_dataset_spectrogram.npy"]
        data_files = ['trainval_Myo.pt', 'test_Myo.pt']
        s3_folder = 'Myo'

    elif task == 'ninapro':
        data_files = ['ninapro_data.npy', 'ninapro_label.npy']
        s3_folder = 'ninapro'

    else: 
        pass

    for data_file in data_files:
        filepath = os.path.join(download_dir, data_file)
        s3_path = os.path.join(s3_folder, data_file)
        if not os.path.exists(filepath):
            s3.download_file(s3_bucket, s3_path, filepath)    

    return

