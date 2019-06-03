# GCnet-pytorch
This is a pytorch type of block，including Non-local block，Simple Non-local block，GC block and all GC block; refer to paper《GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond》

Experiment Dataset:CIFAR 100

TRAIN--

In terminal:$python train.py --dataset cifar100 --arch XXX.model

| models | Params | FLOPs | Top-1 acc |
| ------ | ------ | ------ | ------ |
| Resnet50 | 23.71M | 1.31G | 78.37% |
| Resnet50 + 1NL | 27.91M | 1.58G | 78.91% |
| Res2next29 + 1SNL | 24.76M | 1.31G | 79.01% |
| Res2next29 + 1GC | 23.97M | 1.31G | 79.47% |
| Res2next29 + all GC | 29.07M | 1.32G | 79.55% |
