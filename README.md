# TensorFlow tutorials

Collection of Jupyter notebooks demonstrating best-practices for using TensorFlow on GPU accelerated hardware. 

* Demonstrate end-to-end GPU accelerated ML workflow using TensorFlow to train various ML models and DNN achitectures.
* Demonstrate distributed, GPU accelerated training capable of scaling to clusters of GPUs.

The notebooks will eventually demonstrate the above by replicating (and hopefully extending!) the results of the NERSC [autoscan](http://portal.nersc.gov/project/dessn/autoscan/) project. I was made aware of the autoscan project after reading a [Medium article](https://medium.com/@dessa_/space-2-vec-fd900f5566) about three software engineers from [Dessa](https://www.dessa.com/) who built a convolutional neural network (CNN), which they dubbed [space2vec](https://github.com/pippinlee/space2vec-ml-code), that was capable of out-performing the original random forest modeling used in the autoscan pipeline.


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
