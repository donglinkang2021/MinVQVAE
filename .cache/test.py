import torch
from lightning import LightningModule, LightningDataModule, Trainer
import hydra
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import os

torch.set_float32_matmul_precision('medium')

def get_latest_checkpoint(checkpoints_dir):
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    checkpoint_files.sort()  # Ensure the files are sorted, you can change the sorting logic if needed
    save_path = os.path.join(checkpoints_dir, checkpoint_files[-1])  # Get the latest checkpoint file
    return save_path

@hydra.main(config_path="configs", config_name="test", version_base=None)
def main(cfg):
    dm:LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    # model:LightningModule = hydra.utils.instantiate(cfg.model)
    model_class: type = hydra.utils.get_class(cfg.model._target_)
    exp_prefix = f"{cfg.log_dir}/{cfg.exp_model_name}"
    save_path = get_latest_checkpoint(f"{exp_prefix}/checkpoints")
    output_path = f"{save_path}/checkpoint_lightning.pt"
    convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)
    
    model = model_class.load_from_checkpoint(
        checkpoint_path=output_path,
        hparams_file=f"{exp_prefix}/hparams.yaml"
    )

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.test(model, dm)

if __name__ == '__main__':
    main()

# python train.py