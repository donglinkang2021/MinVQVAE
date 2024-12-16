# Model

We conducted an experiment using the simplest model for testing, which met our initial expectations. Each time, only one variable was changed, demonstrating that less is more.

We ran VQVAE on ImageNet, then replaced the VQ module in VQVAE with the FFN layer from LLM. The FFN layer was tested with and without dropout. The experimental results are as follows:

- **ffnae**: FFN layer without dropout
- **llmffnae**: FFN layer with dropout
- **amvqvae** : Train the VQVAE with updated embedding parameters to see the effect. One version is amvqvae using argmax.
- **smvqvae**: The second version is smvqvae using softmax.

Summary of experimental results:

| Model                          | Test Loss |
|--------------------------------|-----------|
| exp-vqvae-imagenet-2/2024-12-13-23-06-25 | 0.0124    |
| exp-smvqvae-1/2024-12-14-22-52-48        | 0.0124    |
| exp-ffnae-imagenet-0/2024-12-14-17-44-10 | 0.0132    |
| exp-vqvae-imagenet-0/2024-12-07-22-50-03 | 0.0132    |
| exp-amvqvae-1/2024-12-14-22-51-13        | 0.0138    |
| exp-llmffnae-imagenet-0/2024-12-14-17-45-21 | 0.014     |
| exp-vqvae-imagenet-1/2024-12-13-23-08-15 | 0.0218    |

The results show that the FFN layer with dropout performs better than the FFN layer without dropout. However, the VQ model still outperforms the FFN layer with dropout.

Moreover, the VQ model achieves the same effect as the FFN with only half the number of parameters, so we should indeed choose VQ as our model.

An exciting discovery was made: it turns out that the weights of the previous VQ were never updated. This means that in the previous `VQVAE`, a module used `SoftQuantize`, and the embedding part of this module was never trained. Only the ln part was trained. Despite this, the module's performance was very close or even better than the FFN in the LLM. This suggests that we could directly replace the FFN in the LLM with a normally distributed VQ, achieving excellent results without training, with half the parameters, and similar performance. Fine-tuning would be faster, making it almost unbeatable. Given the generality of the FFN, this approach could be applied in most cases.

However, we also found that the results might be influenced by other factors during training. For example, the previous VQVAE achieved the same effect as SMVQVAE without training, but in some cases, the performance was worse. This could be due to the instability caused by loading the model onto different GPUs during training.

Next steps:

0. load the train model and test it on ImageNet eval and test.
1. Replace the trained FFN part of the llmffn with an untrained VQ part to see the effect.
2. Perform few-shot fine-tuning, allowing only the layernorm part of the quantize module to be slightly trained, or not trained at all, to see the effect.
