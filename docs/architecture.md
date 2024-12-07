# Architecture

Removed the previous VQ + Transformer img2img code section for now, focusing on tuning the VQ part first, no rush to add new features since the previous results were not good.

Reorganized the architecture, requiring the installation of deepspeed:

```bash
pip install -U deepspeed
```

Revised the entire architecture, adjusted configuration files using Hydra, and the next step is to focus on tuning the model part.
