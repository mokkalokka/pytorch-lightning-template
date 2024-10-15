# PyTorch Lightning Template - HPC

This template is designed to get you started with an easy to follow DL pipeline for CIFAR100 image classification using HPC resources. The template is based on PyTorch Lightning which reduces boilerplate code and can serve as a starting point for your own DL project.

The guide is written for NTNU's IDUN cluster but most parts of the guide is also applicable for all SLURM based HPC clusters.


# Prerequisites
This template uses Weights & Biases (wandb) for plotting metrics while training.
Wandb is quite usefull in this scenario as it allows you to monitor the training in a browser while the model is training.

Make a Weights and Biases account at: https://wandb.ai/


# How to run the code on IDUN:

### Log into IDUN (ssh)
```
ssh [USERNAME]@idun-login1.hpc.ntnu.no
```

Other options for connection to IDUN is described here:
https://github.com/jarlsondre/idun_tutorial/blob/main/tutorial.md


### Clone the template from Github:
```
git clone https://github.com/mokkalokka/pytorch-lightning-template.git
cd pytorch-lightning-template
```

### Preparing the environment:
```
module load Anaconda3/2023.09-0
conda env create --file environment.yaml
conda activate pl_env
```


### Logging into wandb
```
wandb login
```

Follow the instructions from the terminal output

### Running the training (slurm):
```
sbatch run.slurm
```

When the job has started you can monitor the training on https://wandb.ai/

### Checking the slurm queue
```
squeue -u [USERNAME]
```


# Interactively checking the model inference (after training)

Setting up an interactive compute session and log in via vscode:
https://github.com/jarlsondre/idun_tutorial/blob/main/tutorial.md#connect-local-visual-studio-code-to-idun

- Open inference.ipynb
- Select the pl_env kernel
- Replace "PATH_TO_WEIGHTS_HERE" with the trained weights path under /checkpoints
- Run blocks


Note: If you wish to modify the code, you can also use this interactive session to debug the code with GPU resources before submitting the job via sbatch.

To the trainer interactively:
```
module load Anaconda3/2023.09-0
conda activate pl_env
python trainer.py
```



# Files descriptions:
- **config.yaml** 

    This file contains all the configurations for the model training.
- **datamodule.py** 

    Defines how the data is handeled: download, splits, transforms, dataloaders etc.
- **trainer.py** 

    Defines the network architecture, training pipeline and logging.
- **inference.ipynb** 

    Used for perform inference on the test set interactively.
- **run.slurm** 

    Used for running training using slurm on IDUN.
- **environment.yaml** 

    The packages needed in order to run the code.


# Read more:
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/starter/introduction.html
- CIFAR100: https://www.cs.toronto.edu/~kriz/cifar.html
- IDUN: https://www.hpc.ntnu.no/idun/
- Weights and Biases: https://wandb.ai/
