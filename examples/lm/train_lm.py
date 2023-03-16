import os
import argparse

import wandb
from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, Partial, tree_reduce
import time

import transformers

from omegaconf import OmegaConf

import sys
sys.path.append('.')

from brachy import structure_util as su
from brachy import  optim
from brachy.nn import functional as F
from brachy.optional_module import optional_module

import c4_loader 

from stacked_attention import StackedAttention

parser = argparse.ArgumentParser(description='Jax C4 LM Training')
parser.add_argument('--config', default='examples/lm/config/c4lm_conf.yaml', type=str, help='config file')
parser.add_argument('--wandb', '-w', action='store_true',
                    help='use wandb logging')

def is_number(tree):
    return tree_reduce(lambda x,y: jnp.logical_and(jnp.logical_and(x, jnp.logical_not(jnp.any(jnp.isnan(y)))), jnp.any(jnp.isfinite(y))) , tree, True)

def main():
    args = parser.parse_args()
    global wandb
    wandb = optional_module(wandb, args.wandb)

    config = OmegaConf.load(args.config)

    # jax random numbers seem to cause the huggingface "fast" tokenizers to complain about the process being forked.
    # I have no idea why... it seems unlikely that the prng is actually forking the process but I have not checked the code or anything.
    # anyway, this is why we use the non-"fast" tokenizer below. 
    # tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    config.model.vocab_size = tokenizer.vocab_size


    print("Initializing model...")
    rng = jax.random.PRNGKey(0)
    model_tree, model_config = StackedAttention(config.model, rng=rng)
    # model is a tuple: (model_tree, model_config)
    
    if config.train.mixed_precision:
        model_tree, model_config = optim.mixed_precision_tree((model_tree, model_config), config.train.mixed_precision_scalar)




    print("Initializing optimizer...")
    optconf = config.train.optimizer
    opt = optim.AdamW(model_tree, lr=1.0, betas=list(optconf.betas), weight_decay=optconf.weight_decay)


    opt_tree, opt_config = optim.process_grads.clip_grads(
        opt,
        clip_value=config.train.optimizer.clip, clip_type='per_coordinate')


    print("Setting up dataloader...")
    train_loader = load_c4_data(config, tokenizer)


    wandb.init(project='hax_c4')
    wandb.config.update(OmegaConf.to_container(config))

    print("Starting training loop...")
    train_loop(
        config,
        opt_tree,
        opt_config,
        model_tree,
        model_config,
        train_loader)



def loss(model_tree, model_config, inputs, targets):
    # inputs, targets = batch

    model_update, outputs = su.apply_tree(model_tree, model_config, inputs)

    cross_entropy = F.softmax_cross_entropy(outputs, targets)

    return model_update, cross_entropy, outputs

def train_step(
    opt_tree,
    opt_config,
    model_tree,
    model_config,
    inputs,
    targets,
    token_count,
    lr_scheduler):


    if model_config.get('mixed_precision', False):
        loss_fn = optim.mixed_precision_loss(loss)
    else:
        loss_fn = loss
    
    # output_num=0 indicates that we differentate the first return value after the model_tree...
    # would it be better to instead make this output_num=1??
    loss_and_grad = su.tree_value_and_grad(loss_fn, output_num=0)

    lr = lr_scheduler(token_count)

    hparams  = {'lr': lr}

    opt_update, model_update, cross_entropy, outputs = su.apply_tree(opt_tree, opt_config, hparams, loss_and_grad, model_tree, model_config, inputs, targets)


    predictions = jnp.argmax(outputs, axis=-1)

    correct = jnp.sum(predictions == targets)

    log_data = {
        'loss': cross_entropy,
        'correct': correct,
        'lr_schedule': lr
    }

    return opt_update, model_update, log_data

def get_lr_scheduler(config):
    schedule = config.train.optimizer.get('schedule')
    if schedule is None or schedule == 'none':
        lr_scheduler = lambda tokens: 1.0
    elif schedule == 'linear_decay':
        lr_scheduler = lambda tokens: (config.train.max_tokens - tokens)/config.train.max_tokens
    elif schedule == 'cosine_annealing':
        lr_scheduler = lambda tokens: 0.5 * (1.0 + jnp.cos(jnp.pi * tokens/config.train.max_tokens))
    else:
        raise ValueError(f"unknown schedule type: {schedule}")

    warmup_tokens = config.train.optimizer.get('warmup_tokens', 0)

    warmed_up_lr_scheduler = lambda tokens: config.train.optimizer.lr * jnp.minimum(1.0, tokens/warmup_tokens) * lr_scheduler(tokens)

    return warmed_up_lr_scheduler

def train_loop(
    config,
    opt_tree,
    opt_config,
    model_tree,
    model_config,
    train_loader):


    lr_scheduler = get_lr_scheduler(config)

    # JIT compile the train step.
    train_step_jit = su.jit(train_step, donate_argnums=(0,2), static_argnums=(1,3))

    # statistics to track during training.
    token_count = jnp.array(0.0) # this prevents the jit from tracing again in the second round.
    total_correct = 0
    total_loss = 0.0
    running_loss = 0.0
    running_accuracy = 0.0

    # timestamp markers to throttle log frequency and record training speed.
    last_log_time = 0
    last_token_count = 0

    pbar = tqdm(enumerate(train_loader))



    for batch_idx, batch in pbar:

        inputs = jnp.array(batch['input_ids'])
        targets = jnp.array(batch['labels'])

        tokens = jnp.sum(targets != -100)

        # old_opt_state = opt_state
        # old_model_state = model_state
        opt_tree, model_tree, log_data = train_step_jit(opt_tree, opt_config, model_tree, model_config, inputs, targets, token_count, lr_scheduler)
        

        # this step feels hacky... ideally train_step would just return the entire updated tree, but 
        # unfortunately structure trees are not jax types because they hold things like functions and strings
        # and so cannot be jitted.
        # maybe there is a better way to phrase things though? The below seems especially dangerous
        # because the order of the arguments is important and also non-obvious. Maybe the function should be named
        # in a way that makes it clear what to do...
        # options include designing a clearer api for doing the following merge, or coming up with a way
        # to allow returning entire structure trees that doesn't feel too dangerous. For example, if we assume all non jax-type
        # return values are invariant to the jax-type arguments, then it could probably be done...
        # opt_tree = su.merge_trees(opt_tree, opt_update)
    

        # if is_number(log_data['loss']):
        token_count += tokens
        total_correct += log_data['correct']
        total_loss += log_data['loss'].astype(jnp.float32) * tokens
        running_loss = total_loss/token_count
        running_accuracy = total_correct/token_count


        curtime = time.time()
        log_data.update({
            'tokens': token_count,
            'total_correct': total_correct,
            'running_accuracy': total_correct/token_count,
            'running_loss': running_loss,
            'minibatch_accuracy': log_data['correct']/tokens,
            'minibatch_count': batch_idx,
            'examples': batch_idx * config.train.batch_size,
            'tokens_per_sec': (token_count - last_token_count)/(curtime - last_log_time),
        })


        if curtime > last_log_time + config.train.log_freq_seconds:
            last_log_time = curtime
            last_token_count = token_count
            wandb.log(log_data)

        pbar.set_description('Batch: %d, Current loss: %.3f, Running Loss: %.3f | Running Acc: %.3f%% (%d/%d)'
                     % (batch_idx, log_data['loss'], running_loss, 100.*running_accuracy, total_correct, token_count))

        if token_count >= config.train.max_tokens:
            return




def load_c4_data(config, tokenizer, split='train'):
    loader = c4_loader.get_c4_loader_next_token(tokenizer,
                split=split,
                batch_size=config.train.batch_size,
                max_length=config.model.max_input_length,
                pad_to_multiple_of=config.model.max_input_length,
                num_workers=config.train.num_workers)
    return loader




if __name__ == '__main__':
    main()