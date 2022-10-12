# Paper accepted at NeurIPS 2022!

Website: [DMAP](https://amathislab.github.io/dmap/)

## I want to use DMAP in my custom environment!

## I want to reproduce the experiments reported in the paper!

### Setup

Create a conda environment named `dmap` by running :

``` bash
conda env create -f environment.yml
```

WARNING: the library `pybullet_envs` used in this project suffers from an unresolved import error in the file `<anaconda_dir>/envs/dmap/lib/python3.7/site-packages/pybullet_envs/robot_locomotors.py`. Please change line number 1 to

``` python
from pybullet_envs.robot_bases import XmlBasedRobot, MJCFBasedRobot, URDFBasedRobot
```

and line number 6 to

``` python
from pybullet_envs.robot_bases import BodyPart
```

 to solve the import errors.

### Training with SAC - Simple, Oracle, TCN, DMAP

Activate the environment : `conda activate dmap`

To train an agent, run

``` bash
python main_train.py
```

By default, this script is set to train a Simple Walker with sigma = 0.1. To change agent and algorithm, modify
the config_path parameter, redirecting it to a different .json file in the configs folder (e.g., "walker" -> "hopper",
"simple_walker.json" -> "dmap_hopper.json"). Many other parameters, such as the value of sigma, the random seed or the hyperparameters of SAC, can be set in the configuration file.

The script will log to the folder `output/training/<current_date>`. To track the progress, run `tensorboard --logdir output/training/<current_date>`

### Training of RMA

To train a TCN agent to imitate the environment encoder network of Oracle (RMA training procedure), run:

``` bash
python main_rma.py
```

We provide a pretrained Oracle Half Cheetah, sigma = 0.1 and seed = 2, so that the script works out of the box. To change the environment, sigma and random seed, modify the first lines of the script. NB: make sure you have trained an Oracle
agent with the same parameters, and that it is in the correct subfolder of data (similarly to the pretrained Half Cheetah
model).

### Evaluation of the pretrained agents

To evaluate trained agents, run:

``` bash
python main_evaluate.py
```

The script is set to evaluate Oracle Half Cheetah, sigma = 0.1 and seed = 2, which is provided as a pretrained agent. To evaluate the other provided pretrained agent, DMAP Ant sigma = 0.1 and seed = 2, change the script parameters `env_name` from `"half_cheetah"` to `"ant"` and `algorithm` from `"oracle"` to `"dmap"`. It is also possible to produce the ablation results with the same script, in which DMAP is run ignoring the output of the attention encoding network. In this case, the parameter `algorithm` must be set to `"dmap-ne"`

### Performance analysis

The results of the evaluation of all the trained agents are in the folders `data/<agent_name>/performance`. By running the notebook `performance_dataset.ipynb` it is possible to generate the `pickle` files included in the `data` folder. For the analysis of the results, use the notebook `performance_analysis.ipynb`, which generates the tables included in the paper.

### Attention dataset

To create the attention dataset for a single model, run:

``` bash
python main_attention.py
```

The default configuration will generate the attention dataset for DMAP Ant, sigma = 0.1, seed = 2 (available as a pretrained model). To run this script with other configurations, first make sure to have trained the corresponding DMAP agent.


## Reference

Chiappa, A.S., Vargas, A.M. and Mathis, A., 2022. [DMAP: a Distributed Morphological Attention Policy for Learning to Locomote with a Changing Body. arXiv preprint arXiv:2209.14218.](https://arxiv.org/abs/2209.14218)

```
@article{chiappa2022dmap,
  title={DMAP: a Distributed Morphological Attention Policy for Learning to Locomote with a Changing Body},
  author={Chiappa, Alberto Silvio and Vargas, Alessandro Marin and Mathis, Alexander},
  journal={arXiv preprint arXiv:2209.14218},
  year={2022}
}
```
