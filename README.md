# TensorFlow tutorials

Collection of Jupyter notebooks demonstrating best-practices for using TensorFlow on GPU accelerated hardware. 

* Demonstrate end-to-end GPU accelerated ML workflow using TensorFlow to train various DNN achitectures.
* Demonstrate distributed, GPU accelerated training capable of scaling to clusters of GPUs.

## Using Conda

Create the environment...

```bash
$ conda env create -f environment.yml
```

...then activate the environment...

```bash
$ source activate $(head -1 $environment.yml | cut -d' ' -f2)
```

...then launch the Jupyter notebook server.

```bash
$ jupyter notebook
```
