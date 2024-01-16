# Distributed computing on Ray clusters: recipes

This repo is a starter for learning how to use Ray for distributed computing on both CPU anD GPUs

## Introduction 

This repo contains a collection of small demos for learning to run python tasks in a distributed mode on Ray clusters, specifically on virtual machines in the Azure Cloud.

Running distributed applications on Ray clusters definitely has some learning curve. The intend of this repo is to provide the user with some working examples to start rapidly and reasonably intuitively using this technology.

The templates here give general computing examples, although Ray comes with specialized sub-packages dedicated for data or machine learning applications.

Ray has its dedicated documentation and tutorials, on which this repo is heavily based. The repo is meant to shortcut to rapid start for new users. References are given in the bottom of this document.

## 1. Installing Ray locally

Locally, you can install the default Ray package like this (e.g. in your conda environment): 

```pip install -U "ray[default]"```

## 2. Configuration of a Ray cluster 

As said in the intro, the Ray clusters we intend to build here will live on VMs in Azure. There will be 2 types of configurations, for CPU or for GPU computing. The configurations are in the `configs/` folder.

To launch a cluster using the selected configuration file: 

```ray up -y config.yaml```

(or `ray up -y config.yaml --no-config-cache`, when you restart from scratch a deleted cluster, to avoid "remembering" any configuration of the previous cluster).

And to turn down the cluster: 

```ray down -y config.yaml```

The configuration parameter `cache_stopped_nodes: False` would completely erase the cluster when turning it down, otherwise the VMs of the cluster will remain, in stopped mode.

When started, the cluster can be monitored in a dashboard:

```ray dashboard config.yaml```

[Notes for the cluster update](https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-cli.html#updating-an-existing-cluster-ray-up)

- if the changes don’t require a restart, pass --no-restart to the update call.

- if you want to force re-generation of the config to pick up possible changes in the cloud environment, pass --no-config-cache 

- if you want to skip the setup commands and only restart all nodes, pass --restart-only

### 2.1 Configuration for CPU computing

See example in `configs/config-vm-cpu.yaml`

### 2.2 Configuration for GPU computing

See example in `configs/config-vm-gpu.yaml`

### 2.3 Hybrid case

Here we want to develop the case where the head node (on-demand) is a CPU-only VM, while the worker nodes (spot instances) are GPU-enabled. See `configs/config_hybrid.yaml`

### 2.4 Using Docker images on VMs

Here the Ray cluster is built out of a Docker image, either directly supplied with by Ray dockerhub, or using an image based on it. For a simple example on CPUs, see `configs/config-vm-cpu-docker.yaml`

TODO: describe the case of docker image with GPU support

## 3. Examples of applications

We will start by a distributed application on CPUs, see `scripts/test_multi_workers_cpu.py`

Companion files (modules, etc) can be sent to the Ray cluster using the `rsync` command, like this:

```ray rsync_up config.yaml '/local_home_directory/file.py' '/target_home_directory/directory/' ```

Then the task can be submitted to the cluster, like this (in verbose mode):

```ray submit config.yaml your_code.py -vvvv```


## 4. Useful commands

To monitor the GPU within the docker image of the Ray cluster (or in the VM ir running on the VM directly): `watch -d -n 0.1 nvidia-smi`

## To-Dos, improvements suggested

DONE:

- Upgrade Ray to 2.9.0 and test that working

- DS VMs: migration to empty VMs: it could be a good thing to install Miniconda in an empty machine instead of using these pre-configured, heavy DS machines with 128Gb disk (most of installed libraries/tools are unecessary)

- Use docker images instead of installing directly in the VM, as that will allow to accelerate the creation of the nodes. Try this on CPU example first

- create a custom docker image on top of the Ray docker image, where we install custom libraries and dependencies (specific to the application)

- try docker with gpu. For that we need to install Nvidia container toolkit from the Ray cluster configuration file, so that docker can be able to use the GPU using the Nvidia container toolkit.

TODO:

- Test hybrid CPU/GPU case: head should be on-demand CPU VM while worker nodes should be GPU VMs. Start this in `configs/config_hybrid.yaml`

- Try scaling manually the number of nodes up and down. The simplest way for this would be to use a fix size cluster (i.e. without autoscaling) and modify the cluster configuration by increasing/decreasing manually the number of nodes in the config file, before re-running `ray up` command.






Note: I noticed that the T4 GPU (using runtime Standard_NC4as_T4_v3) is runs very fast in batches by Pytorch... So it could be that the slowing down factor is not the GPU itself, but the CPUs that are feeding it. Should I try with another runtime, with a T4 GPU and 8 CPUs, like Standard_NC8as_T4_v3. The latter has 8 CPUs and 56Gb ram, while Standard_NC4as_T4_v3 has 4 CPUS and 28Gb ram.



## References

- [Excellent general intro to Ray by Jules Damji](https://www.youtube.com/watch?v=LmROEotKhJA&ab_channel=Databricks)

- [Intro by Saturn](https://saturncloud.io/blog/getting-started-with-ray-clusters/)

- [Ray concepts like cluster, head vs worker nodes](https://docs.ray.io/en/latest/cluster/key-concepts.html)

- [Ray educational content repo](https://github.com/ray-project/ray-educational-materials/tree/main)
