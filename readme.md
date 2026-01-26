
cd ~/isaac51
export LD_LIBRARY_PATH=$PWD/python_packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

```bash
cd bluerov_manual
$ISAACSIM_PATH/python.sh scripts/manual_control.py headless=false env.num_envs=1
```

train:
$ISAACSIM_PATH/python.sh scripts/train_ppo_behavior.py headless=true task.env.num_envs=1 save_interval=100

resume training from a checkpoint:
$ISAACSIM_PATH/python.sh scripts/train_ppo_behavior.py headless=true task.env.num_envs=64 algo.checkpoint_path=/path/to/ppo_behavior_iter_100.pt


Test
$ISAACSIM_PATH/python.sh scripts/play_ppo_behavior.py headless=false checkpoint_path=/home/md-ether/MarineGym-main/bluerov_manual/checkpoints/ppo_behavior_final.pt
