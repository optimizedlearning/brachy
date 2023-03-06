# language modeling example

A simple transformer architecture for language modeling with next token prediction. Compared with the resnet example, in this code we use the decorators `organized_init_with_rng` and `organized_apply` to remove some more boilerplate from the modeling code.


To run on SCC, from the root `hax` directory, run `python examples/lm/train_lm.py`. You can submit a batch job with `qsub run_lm.sh`. The submit script uses wandb
for logging, so make sure you have run `wandb init` in the `hax` directory already to set up your environment.