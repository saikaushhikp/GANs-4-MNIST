# GANs 4 MNIST

An implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for generating 28×28 MNIST digit images. This project demonstrates stable GAN training with modern techniques including label smoothing, DCGAN weight initialization, and gradient detachment strategies.

## Features

- **DCGAN Architecture**: Generator and Discriminator networks with transpose/standard convolutions
- **Stability Optimizations**: Label smoothing, label noise injection, and proper gradient flow management
- **Progress Visualization**: Fixed noise sampling every epoch to monitor training quality
- **Auto-calibrated Discriminator**: Discriminator automatically computes flattened feature dimensions via dummy forward pass
- **Pre-trained Models**: Saved model weights in `gan_outputs/`

## Quick Start

### Prerequisites

- Python 3.7+
- PyTorch with CUDA support (optional but recommended)
- torchvision, numpy, matplotlib, tqdm

### Installation

```bash
git clone https://github.com/saikaushhikp/GANs-4-MNIST.git
cd GANs-4-MNIST
pip install torch torchvision numpy matplotlib tqdm
pip install torchsummary  # Optional
```

### Running Training

```bash
python3 run.py
```

The script will auto-download MNIST, train for 100 epochs, save samples to `gan_outputs/`, and output model weights.

## Project Structure

```
.
├── run.py                      # Training orchestration
├── utilizations.py             # Models, loss functions, visualization
└── gan_outputs/                # Generated samples & model weights
```

## Architecture

### Generator
- **Input**: 100D noise vector
- **Pipeline**: ConvTranspose2d chain (256→128→64→1 channels)
- **Output**: 28×28 images (center-cropped from 32×32)
- **Normalization**: BatchNorm + ReLU + final Tanh

### Discriminator
- **Input**: 28×28 images
- **Pipeline**: Conv2d downsampling (1→64→128→256)
- **Output**: Binary classification logit

## Hyperparameters

| Parameter | Default | 
|-----------|---------|
| `batch_size` | 128 |
| `noise_dim` | 100 |
| `lr` | 2e-4 |
| `EPOCHS` | 100 |
| `smoothing` | 0.9 |
| `noisy_labels` | 0.05 |

## Training Insights

- **Healthy training**: D and G losses oscillate with downward trend
- **D loss $\to$ 0**: Discriminator overfitting (increase label noise)
- **Mode collapse**: Check `gan_outputs/` for sample diversity; if absent after 20+ epochs, increase noise injection
- **Data pipeline**: 2 workers + pin_memory=True for GPU; set num_workers=0 on single-core systems

## Loss Functions

Both use `BCEWithLogitsLoss` for stability. **Important**: Do NOT use raw `BCELoss` with logits.

```python
discriminator_loss(logits_real, logits_fake, smoothing=0.9, noisy_labels=0.05)
generator_loss(logits_fake)
```

## Model Customization

### Adjusting Output Size
1. Modify ConvTranspose2d in `get_gen_block()`
2. Update crop in `Generator.forward()`: `out[:, :, top:top+SIZE, left:left+SIZE]`
3. Adjust Discriminator Conv2d symmetrically

### Adding Architecture Depth
- Insert additional `get_gen_block()` or `get_disc_block()` calls
- Keep channel counts as powers of 2

## Weight Initialization

Via `weights_init()` following DCGAN specs:
- **Conv/Linear**: N(0, 0.02)
- **BatchNorm**: N(1, 0.02) for weights, 0 for bias

## References

- Radford et al. (2015): DCGAN architecture
- Salimans et al. (2016): GAN training stability techniques
- LeCun et al.: MNIST dataset

## License

See [LICENSE](LICENSE) for details.

