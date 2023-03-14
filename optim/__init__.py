
import optim.process_grads as process_grads
import optim.sgd as sgd
import optim.mixed_precision as mixed_precision
# import optim.random_scaling as random_scaling
import optim.adamw as adamw

from optim.adamw import AdamW
from optim.sgd import SGD


from optim.mixed_precision import mixed_precision_loss, mixed_precision_tree




def get_model(opt_tree):

    # breadth-first-search through the submodules until we find one with the key 'model_to_optimize'.

    submodules_queue = [opt_tree]

    queue_index = 0

    while queue_index != len(submodules_queue):
        node = submodules_queue[queue_index]

        for name, submodule in node['submodules'].items():
            if name == 'model_to_optimize':
                return submodule
            submodules_queue.append(submodule)
        
        queue_index += 1

    return None

    

    
