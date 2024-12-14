import torch
from minvqvae.models.core.quantize import (
    SoftmaxQuantize, # Epoch 99 mse loss: 1.0444517135620117 # codebook mse diff 0.0072
    ArgmaxQuantize, # Epoch 99 mse loss: 1.8672080039978027 # codebook mse diff 0.0085
    SoftQuantize, # Epoch 99 mse loss: 1.0682817697525024 # codebook mse diff 0
    SimpleQuantize, # Epoch 99 mse loss: 2.0118091106414795 # codebook mse diff 0
)
from tabulate import tabulate
torch.manual_seed(42)
B, T, D = 32, 512, 64
vocab_size = 32
model = SoftmaxQuantize(vocab_size, D)
criterion = torch.nn.MSELoss()
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

codebook_before_training = model.embd.weight.clone()
layernorm_weight_before_training = model.ln.weight.clone()
layernorm_bias_before_training = model.ln.bias.clone()
for epoch in range(epochs):
    input = torch.randn(B, T, D, requires_grad=True)
    target = torch.randn(B, T, D)
    quantize, idxs = model(input)
    loss = criterion(quantize, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} loss: {loss.item()}")
codebook_after_training = model.embd.weight.clone()
layernorm_weight_after_training = model.ln.weight.clone()
layernorm_bias_after_training = model.ln.bias.clone()

# codebook diff
codebook_diff = (codebook_before_training - codebook_after_training).pow(2).mean().item()
# layernorm diff
layernorm_weight_diff = (layernorm_weight_before_training - layernorm_weight_after_training).pow(2).mean().item()
layernorm_bias_diff = (layernorm_bias_before_training - layernorm_bias_after_training).pow(2).mean().item()
# get mean and norm of the codebook before and after training
codebook_before_mean = codebook_before_training.mean().item()
codebook_before_norm = codebook_before_training.norm().item()
codebook_after_mean = codebook_after_training.mean().item()
codebook_after_norm = codebook_after_training.norm().item()
layernorm_weight_before_mean = layernorm_weight_before_training.mean().item()
layernorm_weight_before_norm = layernorm_weight_before_training.norm().item()
layernorm_weight_after_mean = layernorm_weight_after_training.mean().item()
layernorm_weight_after_norm = layernorm_weight_after_training.norm().item()


# Table 1: Model and final loss
model_metrics = {
    "Model": model.__class__.__name__,
    "Epoch 99 mse loss": loss.item()
}
print(tabulate(model_metrics.items(), headers=["Metric", "Value"], tablefmt="fancy_grid"))

# Table 2: Differences
diff_metrics = {
    "codebook mse diff": codebook_diff,
    "layernorm weight mse diff": layernorm_weight_diff,
    "layernorm bias mse diff": layernorm_bias_diff
}
print(tabulate(diff_metrics.items(), headers=["Metric", "Value"], tablefmt="fancy_grid"))

# Table 3: Values before and after training
value_metrics = {
    "Metric": ["codebook mean", "codebook norm", "layernorm weight mean", "layernorm weight norm"],
    "Before": [codebook_before_mean, codebook_before_norm, layernorm_weight_before_mean, layernorm_weight_before_norm],
    "After": [codebook_after_mean, codebook_after_norm, layernorm_weight_after_mean, layernorm_weight_after_norm]
}
print(tabulate(value_metrics, headers="keys", tablefmt="fancy_grid"))

"""
╒═══════════════════╤════════════════════╕
│ Metric            │ Value              │
╞═══════════════════╪════════════════════╡
│ Model             │ SimpleQuantize     │
├───────────────────┼────────────────────┤
│ Epoch 99 mse loss │ 2.0118091106414795 │
╘═══════════════════╧════════════════════╛
╒═══════════════════════════╤════════════╕
│ Metric                    │      Value │
╞═══════════════════════════╪════════════╡
│ codebook mse diff         │ 0          │
├───────────────────────────┼────────────┤
│ layernorm weight mse diff │ 0.0099669  │
├───────────────────────────┼────────────┤
│ layernorm bias mse diff   │ 0.00885676 │
╘═══════════════════════════╧════════════╛
╒═══════════════════════╤═════════════╤═════════════╕
│ Metric                │      Before │       After │
╞═══════════════════════╪═════════════╪═════════════╡
│ codebook mean         │ -0.00110224 │ -0.00110224 │
├───────────────────────┼─────────────┼─────────────┤
│ codebook norm         │ 44.7185     │ 44.7185     │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight mean │  1          │  0.900167   │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight norm │  8          │  7.20134    │
╘═══════════════════════╧═════════════╧═════════════╛

╒═══════════════════╤════════════════════╕
│ Metric            │ Value              │
╞═══════════════════╪════════════════════╡
│ Model             │ SoftQuantize       │
├───────────────────┼────────────────────┤
│ Epoch 99 mse loss │ 1.0682817697525024 │
╘═══════════════════╧════════════════════╛
╒═══════════════════════════╤════════════╕
│ Metric                    │      Value │
╞═══════════════════════════╪════════════╡
│ codebook mse diff         │ 0          │
├───────────────────────────┼────────────┤
│ layernorm weight mse diff │ 0.00962442 │
├───────────────────────────┼────────────┤
│ layernorm bias mse diff   │ 0.00931796 │
╘═══════════════════════════╧════════════╛
╒═══════════════════════╤═════════════╤═════════════╕
│ Metric                │      Before │       After │
╞═══════════════════════╪═════════════╪═════════════╡
│ codebook mean         │ -0.00110224 │ -0.00110224 │
├───────────────────────┼─────────────┼─────────────┤
│ codebook norm         │ 44.7185     │ 44.7185     │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight mean │  1          │  0.901897   │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight norm │  8          │  7.21518    │
╘═══════════════════════╧═════════════╧═════════════╛

╒═══════════════════╤════════════════════╕
│ Metric            │ Value              │
╞═══════════════════╪════════════════════╡
│ Model             │ ArgmaxQuantize     │
├───────────────────┼────────────────────┤
│ Epoch 99 mse loss │ 1.8672080039978027 │
╘═══════════════════╧════════════════════╛
╒═══════════════════════════╤════════════╕
│ Metric                    │      Value │
╞═══════════════════════════╪════════════╡
│ codebook mse diff         │ 0.00847359 │
├───────────────────────────┼────────────┤
│ layernorm weight mse diff │ 0.00967632 │
├───────────────────────────┼────────────┤
│ layernorm bias mse diff   │ 0.0085698  │
╘═══════════════════════════╧════════════╛
╒═══════════════════════╤═════════════╤═════════════╕
│ Metric                │      Before │       After │
╞═══════════════════════╪═════════════╪═════════════╡
│ codebook mean         │ -0.00110224 │ -0.00100092 │
├───────────────────────┼─────────────┼─────────────┤
│ codebook norm         │ 44.7185     │ 41.2706     │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight mean │  1          │  0.901634   │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight norm │  8          │  7.21308    │
╘═══════════════════════╧═════════════╧═════════════╛

╒═══════════════════╤════════════════════╕
│ Metric            │ Value              │
╞═══════════════════╪════════════════════╡
│ Model             │ SoftmaxQuantize    │
├───────────────────┼────────────────────┤
│ Epoch 99 mse loss │ 1.0444517135620117 │
╘═══════════════════╧════════════════════╛
╒═══════════════════════════╤════════════╕
│ Metric                    │      Value │
╞═══════════════════════════╪════════════╡
│ codebook mse diff         │ 0.0072401  │
├───────────────────────────┼────────────┤
│ layernorm weight mse diff │ 0.00891771 │
├───────────────────────────┼────────────┤
│ layernorm bias mse diff   │ 0.00596455 │
╘═══════════════════════════╧════════════╛
╒═══════════════════════╤═════════════╤═════════════╕
│ Metric                │      Before │       After │
╞═══════════════════════╪═════════════╪═════════════╡
│ codebook mean         │ -0.00110224 │  0.00259508 │
├───────────────────────┼─────────────┼─────────────┤
│ codebook norm         │ 44.7185     │ 42.5844     │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight mean │  1          │  0.905586   │
├───────────────────────┼─────────────┼─────────────┤
│ layernorm weight norm │  8          │  7.2447     │
╘═══════════════════════╧═════════════╧═════════════╛
"""