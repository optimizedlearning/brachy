



import os
import argparse

import wandb
from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax.tree_util import tree_map, Partial
import time

import transformers

from omegaconf import OmegaConf

import sys

sys.path.append('.')
import structure_util as su
from optim.adamw import AdamW
from optional_module import optional_module
import functional as F
from optional_module import optional_module
import c4_loader 

from stacked_attention import StackedAttention

parser = argparse.ArgumentParser(description='Jax C4 LM Training')
parser.add_argument('--config', default='examples/lm/config/c4lm_conf.yaml', type=str, help='config file')
parser.add_argument('--wandb', '-w', action='store_true',
                    help='use wandb logging')

def main():
    args = parser.parse_args()
    global wandb
    wandb = optional_module(wandb, args.wandb)

    config = OmegaConf.load(args.config)

    # jax random numbers seem to cause the huggingface "fast" tokenizers to complain about the process being forked.
    # I have no idea why... it seems unlikely that the prng is actually forking the process but I have not checked the code or anything.
    # anywy, this is why we use the non-"fast"    
    # tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    # # huggingface seems to recommend making the pad token some english-readable thing like "<|padtoken|>".
    # # This seems bad to me: if the training set is the internet including huggingface documentation, then
    # # the pad token will naturally appear in the training set, and I want to use the pad token as a marker
    # # to not compute the softmax loss on a token, so it is important that it not show up in the training set.
    # # To avoid this, I choose something more bizarre:
    # pad_token_string = '4P$2A|7D_6D-0I]1N/0G'

    # tokenizer.add_special_tokens({'pad_token': pad_token_string})
    # pad_token = tokenizer(pad_token_string)['input_ids'][0]
    
    config.model.vocab_size = tokenizer.vocab_size


    print("Initializing model...")
    rng = jax.random.PRNGKey(0)
    model_tree, model_config = StackedAttention(config.model, rng=rng)
    model_state, model_apply = su.bind_module(model_tree, model_config)


    print("Initializing optimizer...")
    optconf = config.train.optimizer
    opt_state, opt_apply = AdamW(model_state, lr=1.0, beta1=optconf.beta1, beta2=optconf.beta2, weight_decay=optconf.weight_decay)

    print("Setting up dataloader...")
    train_loader = load_c4_data(config, tokenizer)


    wandb.init(project='hax_c4')
    wandb.config.update(OmegaConf.to_container(config))

    print("Starting training loop...")
    train_loop(
        config,
        opt_state,
        model_state,
        train_loader,
        opt_apply,
        model_apply)



def loss(model_state, model_apply, inputs, targets):

    model_state, outputs = model_apply(model_state, inputs)

    cross_entropy = F.softmax_cross_entropy(outputs, targets)

    return model_state, cross_entropy, outputs

def train_step(
    opt_state,
    model_state,
    inputs,
    targets,
    token_count,
    opt_apply,
    model_apply,
    lr_scheduler):

    loss_and_grad = su.state_value_and_grad(loss, output_num=0)

    l_t = lambda s: loss_and_grad(s, model_apply, inputs, targets)

    lr = lr_scheduler(token_count)

    opt_state, model_state, cross_entropy, outputs = opt_apply(opt_state, model_state, l_t, lr=lr)

    predictions = jnp.argmax(outputs, axis=-1)

    correct = jnp.sum(predictions == targets)

    log_data = {
        'loss': cross_entropy,
        'correct': correct,
        'lr_schedule': lr
    }

    return opt_state, model_state, log_data

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
    opt_state,
    model_state,
    train_loader,
    opt_apply,
    model_apply):


    lr_scheduler = get_lr_scheduler(config)

    # JIT the train step.
    # We first wrap the train step in partial in order
    # to capture all the constant arguments that are not JAX types
    # and so would throw an error when jitted.
    # Since these constant types are all functions, we could technically
    # also wrap them in Partials and then use static_argnums in the jit call,
    # but this way we are guarded against inadvertently changing the arguments and
    # retrigging compilation.
    train_step_partial = Partial(
        train_step,
        model_apply=model_apply,
        opt_apply=opt_apply,
        lr_scheduler=lr_scheduler)
    train_step_jit = jax.jit(train_step_partial)

    # statistics to track during training.
    token_count = 0
    total_correct = 0
    running_loss = 0.0

    # timestamp markers to throttle log frequency and record training speed.
    last_log_time = 0
    last_token_count = 0

    pbar = tqdm(enumerate(train_loader))
    for batch_idx, batch in pbar:

        inputs = jnp.array(batch['input_ids'])
        targets = jnp.array(batch['labels'])

        tokens = jnp.sum(targets != -100)

        opt_state, model_state, log_data = train_step_jit(opt_state, model_state, inputs, targets, token_count)

        token_count += tokens
        total_correct += log_data['correct']
        running_loss += (log_data['loss'] - running_loss)*tokens/(token_count)


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

        pbar.set_description('Batch: %d, Running Loss: %.3f | Running Acc: %.3f%% (%d/%d)'
                     % (batch_idx, running_loss, 100.*total_correct/token_count, total_correct, token_count))

        if tokens >= config.train.max_tokens:
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