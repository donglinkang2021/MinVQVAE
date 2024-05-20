# model
model_name = 'VQVAE'

in_channel=3
hid_channel=128
n_res_block=2
n_res_channel=32
embed_dim=64
n_embed=512

model_kwargs = {
    'in_channel': in_channel,
    'hid_channel': hid_channel,
    'n_res_block': n_res_block,
    'n_res_channel': n_res_channel,
    'embed_dim': embed_dim,
    'n_embed': n_embed
}

# dataset
# dataset_name = 'MNIST'
dataset_name = 'CIFAR10'

image_size = 224
data_dir = 'data'
batch_size = 256
num_workers = 4

dataset_kwargs = {
    'data_dir': data_dir,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'image_size': image_size
}

# Training
learning_rate = 3e-4
epochs = 10

# lightning visulization

vis_kwargs = {
    'n_sample': 32,
    'size': image_size,
    'in_channel': in_channel
}