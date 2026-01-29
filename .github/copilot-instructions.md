# MAGE (Masked Generative Encoder) PyTorch Implementation

## Project Overview
This is a PyTorch implementation of MAGE, originally developed in JAX. It unifies generative modeling and representation learning using a masked image modeling approach on VQGAN-quantized tokens.

## Architecture

### Core Components
- **MAGE Model** (`models_mage.py`): Contains `MaskedGenerativeEncoderViT`.
  - Integrates a **frozen VQGAN** (`taming/models/vqgan.py`) to tokenize images.
  - Implements variable masking ratios using truncated normal distributions (`mask_ratio_mu`, `mask_ratio_std`).
  - Uses a ViT-based encoder-decoder architecture.
- **VQGAN** (`taming/`): Adapted from "Taming Transformers". Defined in `taming/models/vqgan.py` and configured via `config/vqgan.yaml`. It is loaded as a `LightningModule` but used as a standard PyTorch module within MAGE.
- **Entry Points**:
  - `main_pretrain.py`: Pre-training loop (Masked Token Prediction).
  - `main_finetune.py`: Fine-tuning for image classification.
  - `main_linprobe.py`: Linear probing evaluation.
- **Engines**: `engine_pretrain.py` and `engine_finetune.py` contain the core training steps (`train_one_epoch`).

### Data Flow
1.  **Input**: Images loaded via `util/datasets.py` (ImageNet structure).
2.  **Tokenization**: Images $\rightarrow$ VQGAN Encoder $\rightarrow$ Discrete Tokens.
3.  **Masking**: Tokens are masked based on sampled ratios.
4.  **Modeling**: Masked tokens $\rightarrow$ ViT Encoder $\rightarrow$ ViT Decoder $\rightarrow$ Predicted Tokens.
5.  **Loss**: Cross-entropy on masked tokens + Reconstruction loss (implicit in token prediction).

## Workflows

### Training
Training scripts rely heavily on `torch.distributed.launch`.
**Pre-training Example**:
```bash
python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
    --model mage_vit_base_patch16 \
    --batch_size 64 \
    --output_dir ./output_dir \
    --data_path /path/to/imagenet
```

### Inference / Generation
Unconditional generation uses `gen_img_uncond.py`:
- Requires a pre-trained checkpoint (`--ckpt`).
- Iterative decoding process (`--num_iter`).

## Development & Conventions

### Dependencies & Compatibility
- **Strict Versions**: The codebase is sensitive to versions. verify `timm==0.3.2` and Torch versions (originally 1.7.1) if encountering shape/compat issues.
- **JAX Port**: Arguments and logic often mirror the original JAX implementation.
- **Config**:
  - Model params are passed via `argparse` in `main_*.py`.
  - VQGAN params are loaded from `config/vqgan.yaml` using `OmegaConf`.

### Key Coding Patterns
- **Distributed Training**: Uses `util.misc` for `NativeScalerWithGradNormCount` and `MetricLogger`. All main scripts assume distributed context (even on single node).
- **Transforms**: `util/datasets.py` customizes `timm` transforms. Note specific `scale` params.
- **Frozen Components**: When modifying `MaskedGenerativeEncoderViT`, remember `self.vqgan` parameters are explicitly frozen (`requires_grad = False`).

### Common Tasks
- **Adding a new model variant**: Update `models_mage.py` (add `@register_model` decorator if using registry, or just class) and `main_*.py` args.
- **Debugging Data**: Check `util/datasets.py`. Ensure `IMAGENET_DIR` has standard `train/` and `val/` subfolders.
