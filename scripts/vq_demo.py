import torch
from minvqvae.core import (
    SoftmaxQuantize,
    ArgmaxQuantize,
    SoftQuantize,
    SimpleQuantize,
)
from tabulate import tabulate
torch.manual_seed(42)
B, T, D = 32, 512, 64
n_embed = 32
epochs = 100
lr = 1e-3

quantizers = [SoftmaxQuantize, ArgmaxQuantize, SoftQuantize, SimpleQuantize]
results = []

for quantizer_class in quantizers:
    print(f"Training {quantizer_class.__name__}...")
    model = quantizer_class(n_embed, D)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    codebook_before_training = model.embd.weight.clone()
    layernorm_weight_diff = "N/A"
    layernorm_bias_diff = "N/A"
    # For models that might not have layernorm, handle AttributeError
    try:
        layernorm_weight_before_training = model.ln.weight.clone()
        layernorm_bias_before_training = model.ln.bias.clone()
    except AttributeError:
        layernorm_weight_before_training = None
        layernorm_bias_before_training = None

    for epoch in range(epochs):
        input_tensor = torch.randn(B, T, D, requires_grad=True)
        target = (input_tensor + torch.randn(B, T, D) * 0.1).detach()
        quantize, idxs = model(input_tensor)
        loss = criterion(quantize, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == epochs -1 : # print last epoch loss
            print(f"Epoch {epoch} loss: {loss.item()}")

    codebook_after_training = model.embd.weight.clone()
    codebook_diff = (codebook_before_training - codebook_after_training).pow(2).mean().item()
    
    final_loss = loss.item()

    current_result = {
        "Model": model.__class__.__name__,
        "Final MSE loss": final_loss,
        "Codebook MSE diff": codebook_diff,
        "Layernorm weight MSE diff": "N/A", # Default to N/A
        "Layernorm bias MSE diff": "N/A"    # Default to N/A
    }
    
    if layernorm_weight_before_training is not None:
        layernorm_weight_after_training = model.ln.weight.clone()
        layernorm_bias_after_training = model.ln.bias.clone()
        layernorm_weight_diff = (layernorm_weight_before_training - layernorm_weight_after_training).pow(2).mean().item()
        layernorm_bias_diff = (layernorm_bias_before_training - layernorm_bias_after_training).pow(2).mean().item()
        current_result["Layernorm weight MSE diff"] = layernorm_weight_diff
        current_result["Layernorm bias MSE diff"] = layernorm_bias_diff
        

    results.append(current_result)

    # Optional: Detailed print for each model (can be removed if only summary is needed)
    print(f"Finished training {model.__class__.__name__}")
    print(f"  Final MSE loss: {final_loss}")
    print(f"  Codebook MSE diff: {codebook_diff}")
    print(f"  Layernorm weight MSE diff: {layernorm_weight_diff}")
    print(f"  Layernorm bias MSE diff: {layernorm_bias_diff}")
    # The layernorm diffs are printed above if applicable
    print("-" * 30)


# Print summary table
print("\nSummary of Quantization Methods:")
headers = ["Model", "Final MSE loss", "Codebook MSE diff", "Layernorm weight MSE diff", "Layernorm bias MSE diff"]
table_data = [[result.get(header, "N/A") for header in headers] for result in results]
print(tabulate(table_data, headers=headers, tablefmt="github", floatfmt=".4f"))

"""
| Model           |   Final MSE loss |   Codebook MSE diff |   Layernorm weight MSE diff |   Layernorm bias MSE diff |
|-----------------|------------------|---------------------|-----------------------------|---------------------------|
| SoftmaxQuantize |           0.8332 |              0.0080 |                      0.0085 |                    0.0053 |
| ArgmaxQuantize  |           1.4309 |              0.0084 |                      0.0101 |                    0.0085 |
| SoftQuantize    |           0.8693 |              0.0000 |                      0.0100 |                    0.0093 |
| SimpleQuantize  |           1.5104 |              0.0000 |                      0.0100 |                    0.0085 |
"""