# MinVQVAE

Minimal Discrete Variational Autoencoder (VQ-VAE) implementation in PyTorch.

> Beyond VQ-VAE:2, get crazy performance.

## Performance


- 777K VQVAE with `SoftQuantize` of my implementation
- get `0.0002233` MSE Loss on CIFAR-10 test set
- just training 10 epochs cost 11 min on 2x RTX4090 GPUs

| Action                                                                  |  Mean duration (s)  |  Num calls             |  Total time (s)       |  Percentage %         |
|-------------------------------------------------------------------------|---------------------|------------------------|------------------------|-----------------------|
| Total                                                                   |  -                  |  32294                 |  663.84               |  100 %                |
| run_training_epoch                                                      |  56.897             |  10                    |  568.97               |  85.708               |
| [Strategy]DeepSpeedStrategy.validation_step                             |  1.4723             |  222                   |  326.84               |  49.235               |
| run_training_batch                                                      |  0.22579            |  790                   |  178.37               |  26.869               |
| [LightningModule]VQVAELightning.optimizer_step                          |  0.22568            |  790                   |  178.29               |  26.857               |
| [Strategy]DeepSpeedStrategy.backward                                    |  0.20351            |  790                   |  160.77               |  24.218               |
| [_EvaluationLoop].val_next                                              |  0.15036            |  222                   |  33.38                |  5.0284               |
| [Strategy]DeepSpeedStrategy.test_step                                   |  1.5457             |  20                    |  30.914               |  4.6569               |
| [Strategy]DeepSpeedStrategy.batch_to_device                             |  0.02881            |  1032                  |  29.732               |  4.4788               |
| [LightningModule]VQVAELightning.transfer_batch_to_device                |  0.02867            |  1032                  |  29.587               |  4.457                |
| [Strategy]DeepSpeedStrategy.training_step                               |  0.018908           |  790                   |  14.937               |  2.2501               |