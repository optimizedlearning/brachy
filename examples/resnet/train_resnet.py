'''Train CIFAR10 with Jax.'''

# modified from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py


import sys
# I'll fix this later once I actually understand the python import system...
sys.path.append('.')

import torch

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18, PreActResNet18
from brachy import structure_util as su
from brachy.nn import functional as F
from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, Partial

from brachy.optim.sgd import SGD

from brachy.optional_module import optional_module
import wandb



parser = argparse.ArgumentParser(description='Hax CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'preactresnet18'])
parser.add_argument('--wandb', '-w', action='store_true',
                    help='use wandb logging')


def main():
    args = parser.parse_args()


    global wandb
    wandb = optional_module(wandb, args.wandb)

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
        model_tree, model_config = ResNet18(rng)
    elif args.arch == 'preactresnet18':
        model_tree, model_config = PreActResNet18(rng)


    opt_tree, opt_config = SGD(model_tree, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train_step_jit = su.jit(train_step, donate_argnums=(0, 2), static_argnums=(1,3))
    test_step_jit = su.jit(test_step, static_argnums=1)


    wandb.init(project="jax_resnet")
    wandb.config.update(args)
    for epoch in range(200):
        opt_tree, model_tree = train_epoch(epoch, opt_tree, opt_config, model_tree, model_config, trainloader, train_step_jit)
        test(epoch, model_tree, model_config, testloader, test_step_jit)



def cosine_annealing(epoch, max_epochs, eta_max, eta_min):
    return (1 + jnp.cos(jnp.pi * epoch / max_epochs)) * (eta_max - eta_min) * 0.5 + eta_min


def loss(model_tree, model_config, inputs, targets):

    new_tree, output = su.apply_tree(model_tree, model_config, inputs)

    cross_entropy = F.softmax_cross_entropy(output, targets)
    return new_tree, output, cross_entropy


def train_step(opt_tree, opt_config, model_tree, model_config, inputs, targets, epoch, eta_max=1.0, eta_min=0.0, max_epochs=200):


    loss_and_grad = su.tree_value_and_grad(loss, output_num=1)

    lr = cosine_annealing(epoch, max_epochs, eta_max, eta_min)

    hparams = {
        'lr': lr
    }

    opt_tree, model_tree, output, cross_entropy = su.apply_tree(
        opt_tree,
        opt_config,
        hparams,
        loss_and_grad,
        model_tree,
        model_config,
        inputs,
        targets)

    predictions = jnp.argmax(output, axis=-1)
    correct = jnp.sum(predictions == targets)

    log_data = {
        'lr': lr,
        'correct': correct,
        'loss': cross_entropy,
    }

    return opt_tree, model_tree, log_data


def test_step(model_tree, model_config, batch):
    inputs, targets = batch
    _, output, cross_entropy = loss(model_tree, model_config, inputs, targets)
    predictions = jnp.argmax(output, axis=-1)
    correct = jnp.sum(predictions == targets)

    return cross_entropy, correct


# Training
def train_epoch(epoch, opt_tree, opt_config, model_tree, model_config, trainloader, train_step_jit):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    total_loss = 0
    batches = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        # mildly annoying recast since the torchvision transforms only work on pytorch tensors...
        inputs, targets = inputs.detach_().numpy(), targets.detach_().numpy()
        batch = (inputs, targets)

        num_targets = targets.shape[0]

        opt_tree, model_tree, log_data = train_step_jit(opt_tree, opt_config, model_tree, model_config, inputs, targets, epoch)

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

    return opt_tree, model_tree


def test(epoch, model_tree, model_config, testloader, test_step_jit):
    test_loss = 0
    correct = 0
    total = 0
    total_loss = 0

    batches = 0
    pbar = tqdm(enumerate(testloader), total=len(testloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.detach_().numpy(), targets.detach_().numpy()
        batch = (inputs, targets)
        num_targets = targets.shape[0]
        test_loss, batch_correct = test_step_jit(model_tree, model_config, batch)

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


