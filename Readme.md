## Structural Causal Model for 3D Counterfactual Inference

This repository is currently in progress. It can already be used to generate counterfactual images of MR brain images for 3D medical imaging data using deep structural causal models (DSCM).

This work builds on the work done by [Pawlowski et al. (2019)](https://arxiv.org/pdf/2006.06485). The code in this repository is a fork of their code which can be found [here](https://github.com/biomedia-mira/deepscm/tree/master).


### Training Example:
```
python -m deepscm.experiments.synthetic_medical.trainer -e SVIExperiment -m ConditionalVISEM --decoder_type fixed_var --train_batch_size 16 --default_root_dir assets/models/synthetic_medical/ --gpus 0 --latent_dim 128 --pgm_lr 0.0005 --lr 0.0005 --blocks 1,2,4,6,8,8 --ch_multi 10 --num_sample_particles 8
```