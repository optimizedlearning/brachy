# resnet implementation adapted from pytorch implementation in
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import sys
# I'll fix this later once I actually understand the python import system...
sys.path.append('..')
sys.path.append('.')

from structure_util import StateOrganizer
import structure_util as su

import jax
from jax import numpy as jnp
from jax.tree_util import Partial
import rng_util
import nn
from nn import functional as F


def BasicBlock(expansion, in_planes, planes, stride=1, rng=None):
    if rng is None:
        rng = rng_util.split()

    organizer = StateOrganizer()

    organizer.register_buffer('expansion', expansion)
    with rng_util.RNGState(rng):
        organizer.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        organizer.bn1 = nn.BatchNorm2d(planes)
        organizer.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
        organizer.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != organizer.expansion*planes:
            organizer.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, organizer.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(organizer.expansion*planes)
            )
        else:
            organizer.shortcut = nn.Identity()

    return organizer.create_module(BasicBlock_apply)

def BasicBlock_apply(tree, global_config, x):
    organizer = StateOrganizer(tree, global_config)

    out = F.relu(organizer.bn1(organizer.conv1(x)))
    out = organizer.bn2(organizer.conv2(out))
    out += organizer.shortcut(x)
    out = F.relu(out)
    return organizer.get_state(), out




def Bottleneck(expansion, in_planes, planes, stride=1, rng=None):
    if rng is None:
        rng = rng_util.split()

    organizer = StateOrganizer()
    organizer.register_buffer('expansion', expansion)

    with rng_util.RNGState(rng):

        organizer.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        organizer.bn1 = nn.BatchNorm2d(planes)
        organizer.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        organizer.bn2 = nn.BatchNorm2d(planes)
        organizer.conv3 = nn.Conv2d(planes, organizer.expansion *
                                planes, kernel_size=1, bias=False)
        organizer.bn3 = nn.BatchNorm2d(organizer.expansion*planes)

        if stride != 1 or in_planes != organizer.expansion*planes:
            organizer.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, organizer.expansion*planes,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(organizer.expansion*planes)
            )
        else:
            organizer.shortcut = nn.Identity()

    return organizer.create_module(Bottleneck_apply)

def Bottleneck_apply(tree, global_config, x):
    organizer = StateOrganizer(tree, global_config)

    out = F.relu(organizer.bn1(organizer.conv1(x)))
    out = F.relu(organizer.bn2(organizer.conv2(out)))
    out = organizer.bn3(organizer.conv3(out))
    out += organizer.shortcut(x)
    out = F.relu(out)
    return organizer.get_state(), out




def ResNet(block, expansion, num_blocks, num_classes=10, rng=None):


    organizer = StateOrganizer()
    organizer.register_buffer('in_planes', 64)

    def _make_layer(block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(expansion, organizer.in_planes, planes, stride))
            organizer.in_planes = planes * expansion
        return nn.Sequential(*layers)

    if rng is None:
        rng = rng_util.split()
    with rng_util.RNGState(rng):
        organizer.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
        organizer.bn1 = nn.BatchNorm2d(64)
        organizer.layer1 = _make_layer(block, 64, num_blocks[0], stride=1)
        organizer.layer2 = _make_layer(block, 128, num_blocks[1], stride=2)
        organizer.layer3 = _make_layer(block, 256, num_blocks[2], stride=2)
        organizer.layer4 = _make_layer(block, 512, num_blocks[3], stride=2)
        organizer.linear = nn.Linear(512*expansion, num_classes)

    return organizer.create_module(ResNet_apply)



def ResNet_apply(tree, global_config, x):
    # print("x: ",x)
    organizer = StateOrganizer(tree, global_config)

    out = F.relu(organizer.bn1(organizer.conv1(x)))
    out = organizer.layer1(out)
    out = organizer.layer2(out)
    out = organizer.layer3(out)
    out = organizer.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.reshape((out.shape[0], -1))
    out = organizer.linear(out)
    return organizer.get_state(), out



def ResNet18(rng=None):
    return ResNet(BasicBlock, 1, [2, 2, 2, 2], rng=rng)


def ResNet34(rng=None):
    return ResNet(BasicBlock, 1, [3, 4, 6, 3], rng=rng)


def ResNet50(rng=None):
    return ResNet(Bottleneck, 4, [3, 4, 6, 3], rng=rng)


def ResNet101(rng=None):
    return ResNet(Bottleneck, 4, [3, 4, 23, 3], rng=rng)


def ResNet152(rng=None):
    return ResNet(Bottleneck, 4, [3, 8, 36, 3], rng=rng)



# import jax

# rng = jax.random.PRNGKey(0)

# r = ResNet18(rng)