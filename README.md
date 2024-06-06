# SNN-NDOT
This is the PyTorch implementation of paper: NDOT: Neuronal Dynamics-based Online Training for Spiking Neural Networks **(ICML 2024)**. \[[Openreview](https://icml.cc/virtual/2024/poster/33481)\].


## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch, torchvision](https://pytorch.org/)


## Change DIR to your own data dir
```
In `./data/data_loaders.py` and `./datasets/data_loaders.py` 

DIR = {
    'CIFAR10': '/l/users/datasets/CIFAR10',
    'CIFAR100': '/l/users/datasets/CIFAR100',
    'CIFAR10DVS': '/l/users/datasets/CIFAR10DVS',
    'MNIST': '/l/users/datasets/',
    'ImageNet': '/l/users/datasets/',
    'Tiny-ImageNet': '/l/users/datasets/tiny-imagenet-200/'
}
```

## Training
For NDOT$_A$, run as following:

```
python train_cifar_main.py --T 4 --dataset cifar10 --tau 2.0 \
  --model online_spiking_vgg11_ws \
  --output_dir=$checkpoint --tb --autoaug --cutout --drop_rate 0.0 \
  --batch_size 128 --T_max 300 --epochs 300 \
  --optimizer SGD --lr 0.1 --weight_decay 0.0 --gpu_id 0

```


For NDOT$_O$, add the argument `--online_update` as:
```
python train_cifar_main.py --T 4 --dataset cifar10 --tau 2.0 \
  --model online_spiking_vgg11_ws \
  --output_dir=$checkpoint --tb --autoaug --cutout --online_update  --drop_rate 0.0 \
  --batch_size 128 --T_max 300 --epochs 300 \
  --optimizer SGD --lr 0.1 --weight_decay 0.0 --gpu_id 0

```


## Acknowledgement

Some codes for the neuron model and data prepoccessing are adapted from the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repository, and the codes for some utils are from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repository.


## Contact
If you have any questions, please contact Dr. Jiang <jianghaiyan.cn@gmail.com>.


