# TODO

Hope to optimize the repository architecture to facilitate experiments and try different network architectures.

> target1: Optimize the repository architecture and rerun it

- step1: [dataset](./data.md) 

For the previous work, it can be merged into the master branch first.

The second step should be to modify the configuration and rerun it. The executable files and the interdependent packages should be stored separately. At the same time, ensure that each file in the package can be executed independently to maintain code readability and maintainability.

Run `train_scratch.py` first.

For finetuning, we can think about how to proceed later. For now, we can delete it and focus on one direction at a time.
