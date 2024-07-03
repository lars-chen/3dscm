SCM for MR images of MS

This repository holds code to generate counterfactual images of MR brain images for people with (and without) MS using a structural causal model (SCM) built in Pyro.

This code was used to generate the counterfactual images for the thesis project.... XXX

This work builds on the work of Pawlowski, Castro, and Glocker [2]. The code in this repository is a fork of their code which can be found here.


Train:

python -m deepscm.experiments.synthetic_medical.trainer -e SVIExperiment -m ConditionalVISEM --decoder_type fixed_var --train_batch_size 2 --default_root_dir assets/models/synthetic_medical/ --gpus 0 --latent_dim 128 --pgm_lr 0.0005 --lr 0.0005 --blocks 1,2,4,6,8,8 --ch_multi 10 --gpus 3 --num_sample_particles 8
