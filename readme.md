
cd ~/isaac51
export LD_LIBRARY_PATH=$PWD/python_packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

```bash
cd bluerov_manual
$ISAACSIM_PATH/python.sh scripts/manual_control.py headless=false env.num_envs=1
```

train:
$ISAACSIM_PATH/python.sh scripts/train_ppo_behavior.py headless=false task.env.num_envs=1 save_interval=100

resume training from a checkpoint:
$ISAACSIM_PATH/python.sh scripts/train_ppo_behavior.py headless=true task.env.num_envs=64 algo.checkpoint_path=/path/to/ppo_behavior_iter_100.pt


Test
$ISAACSIM_PATH/python.sh scripts/play_ppo_behavior.py headless=false checkpoint_path=/home/md-ether/MarineGym-main/bluerov_manual/checkpoints/ppo_behavior_final.pt






# 📘 README — GPU Usage on IDUN Cluster (Isaac Sim / CUDA Jobs)

## 1. See Available GPU Types and Nodes

### Show GPU types per partition

```bash
sinfo -o "%P %G"
```

### Show nodes and their GPU types

```bash
sinfo -o "%P %N %G"
```

### Show node state (free / mixed / allocated)

```bash
sinfo -N -p GPUQ -o "%N %G %t"
```

State meanings:

* `idle`  → fully free
* `mix`   → partially used
* `alloc` → fully occupied

Prefer **idle** or **mix** nodes.

---

## 2. Request an Interactive GPU Node

### Example: 1× H100 GPU

```bash
srun --partition=GPUQ \
     --gres=gpu:h100:1 \
     --cpus-per-task=8 \
     --mem=64G \
     --time=04:00:00 \
     --pty bash
```

### Other GPU types

```bash
--gres=gpu:a100:1
--gres=gpu:v100:1
--gres=gpu:p100:1
```

---

## 3. Confirm GPU Is Assigned

```bash
nvidia-smi
```

You should see exactly **one GPU**.

Check job info:

```bash
echo $SLURM_JOB_ID
hostname
```

---

## 4. List Your Running Jobs

```bash
squeue -u mded
```

---

## 5. Attach to Existing Job (same terminal)

```bash
sattach JOBID.0
```

Example:

```bash
sattach 23984958.0
```

---

## 6. Open a Second Terminal on the Same Node

This is the correct way:

```bash
srun --jobid=JOBID --overlap --cpu-bind=none --pty bash
```

Example:

```bash
srun --jobid=23984958 --overlap --cpu-bind=none --pty bash
```

Now you have **another shell** on the same GPU node.

---

## 7. Verify You Are on Same Node

In both terminals:

```bash
hostname
```

They must match.

---

## 8. Check GPU Usage Across Terminals

```bash
nvidia-smi
```

You should see your running processes.

---

## 9. Run Isaac Sim (GUI Desktop Node)

Inside GPU node:

```bash
cd ~/isaacsim51
apptainer exec --nv isaac-sim_5.1.0.sif ./isaac-sim.sh
```

(Use **no** `--no-window` when in desktop.)

---

## 10. Run Isaac Sim Headless

```bash
apptainer exec --nv isaac-sim_5.1.0.sif ./isaac-sim.sh --no-window
```

---

## 11. Kill a Job

```bash
scancel JOBID
```

---

## 12. Why Jobs Sometimes Wait

Reasons:

* Requested GPU type busy
* Too much memory requested
* Too many CPUs

Try smaller request:

```bash
--cpus-per-task=4
--mem=32G
```

---

## 13. Quick GPU Health Checks

```bash
nvidia-smi
free -h
df -h
```

---

## 14. Best Practice for Isaac Sim

Recommended:

```
GPU: h100 or a100
CPUs: 8–16
Memory: 64G–128G
Time: 4:00:00+
```

---

## 15. Using Web Desktop (Easiest GUI)

Use cluster **Desktop App**:

* Partition: GPUQ
* GPU type: h100
* GPUs: 1
* CPUs: 8
* Memory: 64GB

Launch → Open Desktop → Terminal → run Isaac Sim.

This avoids X11 and streaming problems.

---

## 16. Quick Troubleshooting

### No GPU inside container

```bash
apptainer exec --nv image.sif nvidia-smi
```

### Job stuck

```bash
squeue -j JOBID
```

### Wrong GPU type

Cancel and request correct one.

---

## 17. Minimal Command Cheat Sheet

```bash
sinfo -N -p GPUQ -o "%N %G %t"
srun --partition=GPUQ --gres=gpu:h100:1 --pty bash
nvidia-smi
squeue -u mded
sattach JOBID.0
srun --jobid=JOBID --overlap --pty bash
scancel JOBID
```

---


