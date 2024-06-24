python -m deepscm.experiments.synthetic_medical.trainer -e SVIExperiment -m ConditionalVISEM --gpus 3 --default_root_dir assets/models/synthetic_medical/ --decoder_type fixed_var --train_batch_size 4 --max_epochs 10 --pgm_lr 0.001 --lr 0.00006

python -m deepscm.experiments.multiple_sclerosis.tester -c assets/models/multiple_sclerosis/SVIExperiment/ConditionalVISEM/version_76/ --gpus $gpu
