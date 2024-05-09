# Just Cluster It!

## Installation
First create a new conda enviroment and install the requirements. For best visualization you should
have wandb account to see the results and visualizations.

```
conda create -n jci python=3.8
pip install -r requirements.txt
```

After go into the torch-ac directory and install torch-ac:
```
cd torch-ac
pip install -e .
```

To change the vizdoom environment you need to add both config files to the vizdoom directory:
```
cp my_way_home_sparse.wad /home/{user}/anaconda3/envs/jci/lib/python3.8/site-packages/vizdoom/scenarios
cp my_way_home_verysparse.wad /home/{user}/anaconda3/envs/jci/lib/python3.8/site-packages/vizdoom/scenarios
cp my_way_home.cfg  /home/{user}/anaconda3/envs/jci/lib/python3.8/site-packages/vizdoom/scenarios
```

Also copy our gymnasium wrapper so you can see the topdown map:
```
cp base_gymnasium_env.py /home/{user}/anaconda3/envs/jci/lib/python3.8/site-packages/vizdoom/gymnasium_wrapper
```


## Example of use


Train the agent on the `MyWayHome` environment with PPO algorithm:

```
python -m scripts.train --algo ppo --model modelName --no-wandb
``

Note that if you want to log with wandb simply remove the --no-wandb argument. You can change the project name in the scripts/train.py file.

Config for Habitat following...