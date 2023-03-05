# ResNet example

The resnet implementation here is lifted almost verbatime from https://github.com/kuangliu/pytorch-cifar. Compare with the original pytorch modeling code to see
how easy it is to transfer the code over. In fact, in this code we explicitly opted not to use the decorators `organized_init_with_rng` and `organized_apply` which
normally make the transfer even more seamless. This way, the code exposes a little more of the internal workings of Hax.

To train: `python train_resnet.py`. On BU's SCC with a V100 GPU, training takes only roughly 70% of the time that the original pytorch implemention does.