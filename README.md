# TensorFlow tutorials

Collection of Jupyter notebooks demonstrating best-practices for using TensorFlow on GPU accelerated hardware. 

* Demonstrate end-to-end GPU accelerated ML workflow using TensorFlow to train various DNN achitectures.
* Demonstrate distributed, GPU accelerated training capable of scaling to clusters of GPUs.

## Using Conda

Create the environment...

```bash
$ conda env create -f environment-gpu.yml
```

...then activate the environment...

```bash
$ source activate $(head -1 $environment-gpu.yml | cut -d' ' -f2)
```

...then launch the Jupyter notebook server.

```bash
$ jupyter notebook
```

If you don't have access to GPU accelerated hardware then most all of the code should work but you will need to create and activate the Conda environment defined by the `environment-cpu.yml`.
