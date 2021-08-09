#for FEAT_NAME in `cat data/zero_shot/train_feats.config_file.8_random_feats.tsv | sed -e 1d | awk '{print $1}' | sed -e s/\"//g`
for FEAT_NAME in `cat data/zero_shot_deepsea/selected_test_feats.txt|head -n2`
do
	sbatch -J $FEAT_NAME -p gpu --mem 32gb --gres gpu:v100-32gb:1 -c 12  --time 3-13 --wrap "/usr/bin/time -v python -u amber_single_run.py --wd outputs/new_20200919/test_feats/single_run/$FEAT_NAME --feat-name $FEAT_NAME"
done
