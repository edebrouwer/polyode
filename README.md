# Anamnesic Neural Differential Equations with Orthogonal Polynomials Projections

![alt text](https://github.com/edebrouwer/orthopoly/blob/main/Fig1.png)

This is the implementation of the ICLR 2023 paper [Anamnesic Neural Differential Equations with Orthogonal Polynomial Projections](https://arxiv.org/abs/2303.01841)


## Installing the package

The dependencies for the project are in `pyproject.toml`. I recommend you create an environement using conda and poetry.
First create a conda environment.

`conda create -n polyode python=3.9` 

Then install poetry,

`curl -sSL https://install.python-poetry.org | python3 -`

Finally install the project.

`poetry install`

You are all set !


## Training PolyODEs

The experiments showed in the paper happen in two steps. The first steps trains the PolyODE on forecasting. The second step freezes the embeddings learnt by this representation and trains an auxiliary classifier or regressor depeding on the nature of the downstream task.

The models are logged using `wandb` so at the moment you'll need to have an account for logging your runs. You can provide your user name as argument.

### Forecasting. 

To train the PolyODE on forecasting : 

`cd polyode/train_scripts`

`poetry run python train_node.py --model_type=CNODExt --data_type={Lorenz,SimpleTraj,MIMIC} --irregular_rate=0.3 --method=implicit_adams --wandb_user=YOUR_USER_NAME`

The following commands are used for the different datasets:

#### Synthetic

`poetry run python train_node.py --model_type=CNODExt --data_type=SimpleTraj --delta_t=0.05 --extended_ode_mode=true --irregular_rate={0.7,0.8,0.9} --method=implicit_adams --gpus=1`

#### Lorenz

`poetry run python train_node.py --model_type=CNODExt --data_type=Lorenz --Nobs=100 --delta_t=0.05 --extended_ode_mode=true --irregular_rate={0.3,0.4,0.5} --method=implicit_adams --lorenz_dims=2 --mode_96=false --gpus=1`

#### Lorenz96

`poetry run python train_node.py --model_type=CNODExt --data_type=Lorenz --Nobs=100 --delta_t=0.05 --extended_ode_mode=true --irregular_rate={0.3,0.4,0.5} --method=implicit_adams --lorenz_dims=4 --mode_96=true --gpus=1`

#### MIMIC

`poetry run python train_node.py --model_type=CNODExt --data_type=MIMIC --Delta=10 --extended_ode_mode=true --hidden_dim=18  --method=implicit_adams --lorenz_dims=4 --gpus=1 `
### Classification and Regression

To train a model on the resulting embeddings, one can use the `classif.py` script.

For instance:

`poetry run python classif.py --Nobs=100 --data_type=Lorenz --init_sweep_id={the id of the sweep you used for the pre-training part} --lorenz_dims=2 --model_type=CNODExt --pre_compute_ode=True --regression_mode={true,false}`
