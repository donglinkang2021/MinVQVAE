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

An exciting discovery was made: it turns out that the weights of the previous VQ were never updated. This means that in the previous `VQVAE`, a module used `SoftQuantize`, and the embedding part of this module was never trained. Only the ln part was trained. Despite this, the module's performance was very close to the FFN in the LLM. This suggests that we could directly replace the FFN in the LLM with a normally distributed VQ, achieving excellent results without training, with half the parameters, and similar performance. Fine-tuning would be faster, making it almost unbeatable. Given the generality of the FFN, this approach could be applied in most cases.

Next steps:

1. Train the VQVAE with updated embedding parameters to see the effect. One version is amvqvae using argmax.
2. The second version is smvqvae using softmax.
3. Replace the trained FFN part of the llmffn with an untrained VQ part to see the effect.
4. Perform few-shot fine-tuning, allowing only the layernorm part of the quantize module to be slightly trained, or not trained at all, to see the effect.

I have reached the second step, but previously encountered some issues where the entire folder was too large due to storing images, resulting in 22GB of data for each experiment. Therefore, after renaming an experiment, tensorboard takes a long time to load the interface. Tomorrow, I can write a script to uniformly test the performance of each trained model on ImageNet eval and test.
