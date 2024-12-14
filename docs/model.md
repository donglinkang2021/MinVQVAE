# Model

We conducted an experiment using the simplest model for testing, which met our initial expectations. Each time, only one variable was changed, demonstrating that less is more.

We ran VQVAE on ImageNet, then replaced the VQ module in VQVAE with the FFN layer from LLM. The FFN layer was tested with and without dropout. The experimental results are as follows:

- **ffnae**: FFN layer without dropout
- **llmffnae**: FFN layer with dropout

Summary of experimental results:

| Model                | MSE Loss                              |
|----------------------|---------------------------------------|
| VQ                   | 0.013192754238843918 |
| FFN without dropout  | 0.02176494151353836 |
| FFN with dropout     | 0.012446979992091656 |

The results show that the FFN layer with dropout performs better than the FFN layer without dropout. However, the VQ model still outperforms the FFN layer with dropout.

Moreover, the VQ model achieves the same effect as the FFN with only half the number of parameters, so we should indeed choose VQ as our model.
