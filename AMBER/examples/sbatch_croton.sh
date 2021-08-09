#!/usr/bin/env bash
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --partition=gpu
#SBATCH --mem 32g
#SBATCH -c 24

echo 'Just plain run.'

# Load bash config.
source "${HOME}"'/.bashrc'
if [ $? != 0 ]; then
    echo 'Failed sourcing '"${HOME}"'/.bashrc'
    exit 1
fi

# Start up anaconda.
conda activate 'amber'
if [ $? != 0 ]; then
    echo 'Failed to activate conda environment.'
    exit 1
fi

# Change directories.
SRC_DIR='.'
cd "${SRC_DIR}"
if [ $? != 0 ]; then
    echo 'Failed changing directories to '"${SRC_DIR}"
    exit 1
fi

# Run train script.
/usr/bin/time -v python examples/amber_croton.py $1
echo $?

# Deactivate conda.
conda deactivate

