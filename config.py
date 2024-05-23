# model
model_name = 'VQVAE_unmask'
# model_name = 'VQVAE'
# model_name = 'SQATE'
# model_name = 'VQVAE_finetune'

in_channel=3
# in_channel=1
hid_channel=128
n_res_block=2
n_res_channel=32
embed_dim=64
n_embed=512
scale_factor=1

model_kwargs = {
    'in_channel': in_channel,
    'hid_channel': hid_channel,
    'n_res_block': n_res_block,
    'n_res_channel': n_res_channel,
    'embed_dim': embed_dim,
    'n_embed': n_embed,
    'scale_factor': scale_factor
}

transformer_kwargs = {
    'n_embd': embed_dim,
    'n_head': 8,
    'n_layer': 2,
    'block_size': 3600,
    'dropout': 0.1
}

# dataset
# dataset_name = 'MNIST'
# dataset_name = 'CIFAR10'
# dataset_name = 'CelebA'
dataset_name = 'ImageNet'

image_size = 224
# data_dir = '/root/autodl-tmp/'
data_dir = "/root/autodl-tmp/imagenet/"
batch_size = 512
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

# mask

mask_prob = 0.5
patch_size = 32
mask_kwargs = {
    'mask_prob': mask_prob,
    'patch_size': patch_size
}