#!/usr/bin/env bash
#SBATCH --time=4-12:00:00
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --partition=gpu
#SBATCH --mem 96g
#SBATCH -c 32

INDEX=$1
echo 'Just plain run for INDEX='"${INDEX}"

if [ "$SLURM_JOB_ID" == "" ]; then
	SLURM_JOB_ID="test-amber"
fi

# Load bash config.
source "${HOME}"'/.bashrc'
if [ $? != 0 ]; then
    echo 'Failed sourcing '"${HOME}"'/.bashrc'
    exit 1
fi

# Start up conda.
conda activate
if [ $? != 0 ]; then
    echo 'Failed to activate conda.'
    exit 1
fi

# Start up anaconda.
conda activate 'amber-zh'
if [ $? != 0 ]; then
    echo 'Failed to activate conda environment.'
    exit 1
fi

# Change directories.
SRC_DIR='.'
cd "${SRC_DIR}"
mkdir -p "${SRC_DIR}"'/outputs/new_20200919/long_and_dilation.ppo.'"${INDEX}"
if [ $? != 0 ]; then
    echo 'Failed changing directories to '"${SRC_DIR}"
    exit 1
fi

# Remove things on ram disk.
function cleanup {
    echo 'Cleaning up data on ram disk.'
    rm -rf '/dev/shm/'"${SLURM_JOB_ID}"
    if [ $? != 0 ]; then
        echo 'Failed to cleanup after exit.'
        exit 1
    else
        echo 'Finished cleaning up data on ram disk.'
    fi
}

# Set trap for exit to ensure cleanup.
trap cleanup EXIT

# Make ram disk.
echo 'Making ram disk dir.'
#SLURM_JOB_ID='test-amber'
mkdir '/dev/shm/'"${SLURM_JOB_ID}"
if [ $? != 0 ]; then
    echo 'Failed to make dir /dev/shm/'"${SLURM_JOB_ID}"
    exit 1
else
    echo 'Finished making ram disk dir.'
fi

# Copy genome to ram disk.
echo 'Copying hdf5 data to ram disk.'
cp './data/zero_shot_deepsea/train.h5' '/dev/shm/'"${SLURM_JOB_ID}"'/train.h5'
cp './data/zero_shot_deepsea/val.h5' '/dev/shm/'"${SLURM_JOB_ID}"'/val.h5'
if [ $? != 0 ]; then
    echo 'Failed to copy genome to /dev/shm/'"${SLURM_JOB_ID}"'/hg19.encoded.h5'
    exit 1
else
    echo 'Finished copying hdf5.'
fi

# Run train script.
/usr/bin/time -v python -u "${SRC_DIR}"'/zero_shot_nas.real_deepsea.py' \
    --model-space long_and_dilation \
    --ppo \
    --wd "${SRC_DIR}"'/outputs/new_20200919/long_and_dilation.ppo.'"${INDEX}" \
    --config-file "${SRC_DIR}"'/data/zero_shot_deepsea/train_feats.config_file.tsv' \
    --train-file '/dev/shm/'"${SLURM_JOB_ID}"'/train.h5' \
    --val-file '/dev/shm/'"${SLURM_JOB_ID}"'/val.h5' \
    --dfeature-name-file "${SRC_DIR}"'/data/zero_shot_deepsea/dfeatures_ordered_list.txt'
echo $?

# Deactivate conda.
conda deactivate

