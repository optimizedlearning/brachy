'''
implements a stacked causal multi-headed self-attention a la GPT, BERT etc.
'''


import sys
# I'll fix this later once I actually understand the python import system...
sys.path.append('.')

from structure_util import StateOrganizer
import structure_util as su

import jax
from jax import numpy as jnp
from jax.tree_util import Partial
import rng_util
import nn
from nn import functional as F


@su.organized_init_with_rng
def AttentionBlock(organizer, config, rng=None):
    # the decorator will take care of setting and propogating the rng to submodules

    embed_dim = config.embed_dim
    num_heads = config.num_heads
    organizer.self_attention = nn.CausalSelfAttention(embed_dim, num_heads)
    organizer.ln = nn.LayerNorm(embed_dim)
    organizer.expand_fc = nn.Linear(embed_dim, 2 * embed_dim)
    organizer.contract_fc = nn.Linear(2 * embed_dim, embed_dim)

    organizer.set_apply(AttentionBlock_apply)

@su.organized_apply
def AttentionBlock_apply(organizer, x):
    y = organizer.ln(x)
    y = organizer.self_attention(y)
    y = F.gelu(y)
    y = organizer.expand_fc(y)
    y = F.gelu(y)
    y = organizer.contract_fc(y)
    y = x + y
    return y

@su.organized_init_with_rng
def StackedAttention(organizer, config, rng=None):
    
    num_layers = config.num_layers
    max_length = config.max_input_length
    embed_dim = config.embed_dim
    vocab_size = config.vocab_size

    organizer.position_embedding = rng_util.normal(shape=(max_length, embed_dim))
    organizer.token_embedding = nn.Embedding(vocab_size, embed_dim)

    organizer.trunk = nn.Sequential(*[
        AttentionBlock(config) for _ in range(num_layers)
    ])

    organizer.head = nn.Linear(embed_dim, vocab_size)

    organizer.set_apply(StackedAttention_apply)


@su.organized_apply
def StackedAttention_apply(organizer, tokens):

    B, T = tokens.shape
    if T > organizer.position_embedding.shape[0]:
        raise TypeError(f'input token size {T} too big!')

    pos_embed = organizer.position_embedding[:T, :]
    tok_embed = organizer.token_embedding(tokens)

    combined_embed = pos_embed + tok_embed

    final_embed = organizer.trunk(combined_embed)

    final_logits = organizer.head(final_embed)

    return final_logits




    



