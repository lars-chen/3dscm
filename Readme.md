## Structural Causal Model for 3D Counterfactual Inference

Abstract: Investigating neurological diseases often involves causal reasoning, such as predicting how brain scans might have differed under alternative treatments. Deep Structural Causal Models (DSCM) enable efficient causal inference using deep learning, showing promise in clinical applications. However, existing DSCMs are largely limited to 2D neuroimaging data and synthetic validation, precluding 3D analysis. We address these gaps by (1) generating a synthetic 3D neuroimaging dataset using a pre-trained diffusion model conditioned on covariates from a ground-truth structural causal model (SCM), and (2) extending DSCMs to 3D via lightweight convolutional neural networks. Our 3D DSCM generates counterfactuals that preserve anatomical structures. When intervening on variables in causal graphs, the model induces anatomical changes in the 3D magnetic resonance images aligned with ground-truth causal effects but underestimates their magnitude, likely due to latent space trade-offs. This work advances 3D causal inference for neuroimaging, providing a benchmark for evaluating counterfactual validity and highlighting challenges in scaling DSCMs to high-dimensional data.


### Training Example:
```
python -m deepscm.experiments.brain_atrophy.trainer -e SVIExperiment -m ConditionalVISEM --decoder_type fixed_var --train_batch_size 16 --default_root_dir assets/models/synthetic_medical/ --gpus 0 --latent_dim 128 --pgm_lr 0.0005 --lr 0.0005 --blocks 1,2,4,6,8,8 --ch_multi 10 --num_sample_particles 8
```
