import os 
import boto3

def download_from_s3(s3_bucket, task, download_dir):
    s3 = boto3.client("s3")
    
    if task == 'smnist':
        data_files = ["s2_mnist.gz"]
        s3_folder = 'spherical'

    if task == 'scifar100':
        data_files = ["s2_cifar100.gz"]
        s3_folder = 'spherical'

    elif task == 'sEMG':
        #data_files = ["saved_evaluation_dataset_test0.npy", "saved_evaluation_dataset_test1.npy",
        #              "saved_evaluation_dataset_training.npy", "saved_pre_training_dataset_spectrogram.npy"]
        data_files = ['trainval_Myo.pt', 'test_Myo.pt']
        s3_folder = 'Myo'

    elif task == 'ninapro':
        data_files = ['ninapro_train.npy', 'ninapro_val.npy', 'ninapro_test.npy',
                      'label_train.npy', 'label_val.npy', 'label_test.npy']
        s3_folder = 'ninapro'

    elif task =='cifar10' or task =='cifar100': 
        return

    elif task == 'audio':
        data_files = ['audio.zip']
        s3_folder = 'audio'

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

    #extract zip if audio
    if task == 'audio' and not os.path.exists(os.path.join(download_dir, 'data')):
        #raise ValueError('Check dir')
        os.mkdir(os.path.join(download_dir,'data'))
        import zipfile
        with zipfile.ZipFile(os.path.join(download_dir, 'audio.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(download_dir, 'data'))

    return

