#!/bin/sh
### Set the job name (for your reference)
#PBS -N languagemodel
### Set the project name, your department code by default
#PBS -P ml.cse
### Set queue priority
###PBS -q high
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=8:ngpus=1:mem=32G:centos=skylake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=05:55:00

#PBS -l software=PYTHON
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR
#job
module () {
	eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
}

module load apps/anaconda/3
source activate ~/tensorflow1
module unload apps/anaconda/3

###mv data/0/xaa data/0/input.txt

###python3 codefiles2/multi-gpu-train.py --data_dir=data --saved_models_dir=saved_models --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=128 --num_gpus=1

mv data/0/input.txt data/0/xaa
mv data/0/xab data/0/input.txt

python3 codefiles2/multi-gpu-train.py --data_dir=data --restore_path=saved_models/0 --saved_models_dir=saved_models --shard=1 --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=512 --num_gpus=1

rm -r languagemodel.*
rm -r training_logs/14
rm -r saved_models/13
rm -r saved_models/initial
mv data/0/input.txt data/0/xae
mv data/0/xaf data/0/input.txt

python3 codefiles2/multi-gpu-train.py --data_dir=data --restore_path=saved_models/14 --saved_models_dir=saved_models --shard=15 --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=128 --num_gpus=1

rm -r languagemodel.*
rm -r training_logs/15
rm -r saved_models/14
rm -r saved_models/initial
mv data/0/input.txt data/0/xaf
mv data/0/xag data/0/input.txt

python3 codefiles2/multi-gpu-train.py --data_dir=data --restore_path=saved_models/15 --saved_models_dir=saved_models --shard=16 --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=128 --num_gpus=1

rm -r languagemodel.*
rm -r training_logs/16
rm -r saved_models/15
rm -r saved_models/initial
mv data/0/input.txt data/0/xag
mv data/0/xah data/0/input.txt

python3 codefiles2/multi-gpu-train.py --data_dir=data --restore_path=saved_models/16 --saved_models_dir=saved_models --shard=17 --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=128 --num_gpus=1

rm -r languagemodel.*
rm -r training_logs/17
rm -r saved_models/16
rm -r saved_models/initial
mv data/0/input.txt data/0/xah
mv data/0/xai data/0/input.txt

python3 codefiles2/multi-gpu-train.py --data_dir=data --restore_path=saved_models/17 --saved_models_dir=saved_models --shard=18 --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=128 --num_gpus=1

rm -r languagemodel.*
rm -r training_logs/18
rm -r saved_models/17
rm -r saved_models/initial
mv data/0/input.txt data/0/xai
mv data/0/xaj data/0/input.txt


python3 codefiles2/multi-gpu-train.py --data_dir=data --restore_path=saved_models/18 --saved_models_dir=saved_models --shard=19 --log_dir=training_logs --rnn_size=2048 --batch_size=128 --seq_length=256 --embedding_size=128 --num_gpus=1

rm -r languagemodel.*
rm -r training_logs/19
rm -r saved_models/18
rm -r saved_models/initial
mv data/0/input.txt data/0/xaj
###mv data/0/xag data/0/input.txt





#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
