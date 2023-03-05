'''Train CIFAR10 with PyTorch.'''

# modified from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
import sys
# I'll fix this later once I actually understand the python import system...
# sys.path.append('../')
# sys.path.append('../../')
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import *
from structure_util import StateOrganizer, bind_module, unbind_module, split_tree, merge_trees
from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, Partial

from optim.sgd import SGD

from optional_module import optional_module
import wandb



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'preactresnet18'])
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--wandb', '-w', action='store_true',
                    help='resume from checkpoint')


def main():
    args = parser.parse_args()


    global wandb
    wandb = optional_module(wandb, args.wandb)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    # we're going to use the pytorch data loader and then transform back from
    # tensor to np array when the data is fetched.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')



    # Model
    print('==> Building model..')
    rng = jax.random.PRNGKey(0)
    if args.arch == 'resnet18':
        net, global_config = ResNet18(rng)
    elif args.arch == 'preactresnet18':
        net, global_config = PreActResNet18(rng)


    state, train_apply = bind_module(net, global_config)
    test_apply = train_apply.bind_global_config({'train_mode': False, 'batch_axis': None})


    sgd_state, sgd_apply = SGD(state, lr=args.lr, momentum=0.9, weight_decay=5e-4)


    train_step_partial = Partial(
        train_step,
        apply=train_apply,
        opt_step=sgd_apply,
        max_epochs=200,
    )


    test_step_partial = Partial(
        test_step,
        apply=test_apply
    )

    train_step_jit = jax.jit(train_step_partial, donate_argnums=(0, 1))
    test_step_jit = jax.jit(test_step_partial)


    wandb.init(project="jax_resnet")
    wandb.config.update(args)
    for epoch in range(start_epoch, start_epoch+200):
        state, sgd_state = train_epoch(epoch, state, sgd_state, trainloader, train_step_jit)
        test(epoch, state, testloader, test_step_jit)



def cosine_annealing(epoch, max_epochs, eta_max, eta_min):
    return (1 + jnp.cos(jnp.pi * epoch / max_epochs)) * (eta_max - eta_min) * 0.5 + eta_min


def loss(state, inputs, targets, apply):

    new_state, output = apply(state, inputs)

    cross_entropy = F.softmax_cross_entropy(output, targets)
    return new_state, output, cross_entropy


def train_step(state, opt_state, inputs, targets, epoch, apply, opt_step, eta_max=1.0, eta_min=0.0, max_epochs=200):


    loss_and_grad = su.state_value_and_grad(loss, output_num=1)
    l_t = lambda s: loss_and_grad(s, inputs, targets, apply)

    lr = cosine_annealing(epoch, max_epochs, eta_max, eta_min)

    opt_state, state, output, cross_entropy = opt_step(opt_state, state, l_t, lr=lr)

    predictions = jnp.argmax(output, axis=-1)
    correct = jnp.sum(predictions == targets)

    log_data = {
        'lr': lr,
        'correct': correct,
        'loss': cross_entropy,
    }

    return state, opt_state, log_data


def test_step(state, inputs, targets, apply):

    _, output, cross_entropy = loss(state, inputs, targets, apply)
    predictions = jnp.argmax(output, axis=-1)
    correct = jnp.sum(predictions == targets)

    return cross_entropy, correct


# Training
def train_epoch(epoch, state, opt_state, trainloader, train_step_jit):#, t_module):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    total_loss = 0
    batches = 0
    pbar = enumerate(trainloader) #
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.detach_().numpy(), targets.detach_().numpy()

        num_targets = targets.shape[0]

        state, opt_state, log_data = train_step_jit(state, opt_state, inputs, targets, epoch)

        cross_entropy = log_data['loss']
        batch_correct = log_data['correct']


        total += num_targets
        batches += 1
        total_loss += cross_entropy
        correct += batch_correct
        

        pbar.set_description('Batch: %d/%d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (batch_idx, len(trainloader), total_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({
        'train/accuracy': correct/total,
        'epoch': epoch,
        'train/loss': total_loss/batches,
        'train/lr': log_data['lr'],
    })

    return state, opt_state


def test(epoch, state, testloader, test_step_jit):
    test_loss = 0
    correct = 0
    total = 0
    total_loss = 0

    batches = 0
    pbar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.detach_().numpy(), targets.detach_().numpy()
        num_targets = targets.shape[0]
        test_loss, batch_correct = test_step_jit(state, inputs, targets)

        total_loss += test_loss
        batches += 1

        total += num_targets
        correct += batch_correct

        pbar.set_description('Batch: %d/%d Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (batch_idx, len(testloader), total_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({
        'test/accuracy': correct/total,
        'epoch': epoch,
        'test/loss': total_loss/batches
    })


if __name__ == '__main__':
    main()


