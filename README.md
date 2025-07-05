# dynamical-diffusion
About Code release for "Dynamical Diffusion: Learning Temporal Dynamics with Diffusion Models" (ICLR 2025)

## Usage

### Training

To train a model (e.g., DyDiff for Turbulence), run:

```bash
cd core; python train_turbulence_dydiff.py --config_file models/turbulence/dydiff_ema_cosine_ratio05_1st_mask_svd_lr1e-4.yaml
```

**Note:** Before running, we need to update the config file with the `root` for data and the `ckpt_path` for VAE model.

### Sampling

To sampling with the trained model, run:

```bash
cd core; python train_turbulence_dydiff.py --config_file models/turbulence/dydiff_ema_cosine_ratio05_1st_mask_svd_lr1e-4.yaml --resume ${model_ckpt} --test
```

This will generate samples in the `logs` directory.

### Evaluation

After generating samples, evaluate them using the following command:

```bash
python core/evaluation/evaluate_turbulence.py --model_output_root logs/turbulence/dydiff_ema_cosine_ratio05_1st_mask_svd_lr1e-4/output_for_evaluation --i3d_model_path ${pretrained_i3d_model}
```
