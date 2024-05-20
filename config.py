in_channel=1
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
image_size = 32

# Training
batch_size = 512
learning_rate = 3e-4
epochs = 10