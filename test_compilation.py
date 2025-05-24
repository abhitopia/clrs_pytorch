import torch
from clrs.dataset import AlgoFeatureDataset, CyclicAlgoFeatureDataset, Algorithm
from clrs.model import Model, ReconstMode
from clrs.processors import Processor
from clrs.utils import tree_map
import pytorch_lightning as pl

pl.seed_everything(42)

torch._dynamo.config.verbose = True

import torch._logging as torch_logging

# Control the logging to help debug compilation issues
torch_logging.set_logs(
#     # dynamo=logging.DEBUG,
#     # inductor=logging.DEBUG,
    recompiles=True,
#     guards=True,
    graph_breaks=True
)

seed = 42
num_batches = 100

algorithms = [Algorithm.dfs, Algorithm.articulation_points]
sizes = [16]

chunk_size = 8
batch_size = 32
static_batch_size = True

algo_datasets = []
for algorithm in algorithms:
    algo_sizes = sizes

    algo_datasets.append(AlgoFeatureDataset(
                                        algorithm=algorithm,
                                        sizes=algo_sizes,
                                        chunk_size=chunk_size,
                                        batch_size=batch_size,
                                        num_batches=num_batches,
                                        seed=seed,
                                        static_batch_size=static_batch_size,
                                        algo_kwargs={}))    
    
dataset = CyclicAlgoFeatureDataset(algo_datasets)
specs = dataset.specs
dl = dataset.get_dataloader(num_workers=0)


hidden_dim = 128
processors_kwargs = {
            "hidden_dim": hidden_dim, 
            "use_ln": True, 
            "nb_heads": 1, 
            "triplet_fts_size": 8,
            "mp_steps": 1
        }


if len(algorithms) == 1:
    # Paper uses triplet GMPNN for single algorithm training
    processor = Processor.triplet_gmpnn(**processors_kwargs)
else:
    # Paper uses triplet MPNN for multi-algorithm training
    processor = Processor.triplet_mpnn(**processors_kwargs)

model = Model(specs=specs,
                processor=processor,
                hidden_dim=hidden_dim,
                encode_hints=True,
                decode_hints=True,
                use_lstm=False,
                hint_reconst_mode=ReconstMode.SOFT,
                hint_teacher_forcing=0.0,
                dropout=0.0)

model.compile()


def get_model_state(model, prev_model_state, is_first, features):
    for algo, batch_is_first in is_first.items():
        if batch_is_first:
            assert prev_model_state[algo] is None
            prev_model_state[algo] = model.init_model_state(algo, features[algo])
        else:
            prev_model_state[algo] = prev_model_state[algo]
    return prev_model_state
 
def set_model_state(prev_model_state, is_last, next_model_state):
    for algo, batch_is_last in is_last.items():
        if batch_is_last:
            prev_model_state[algo] = None
        else:
            prev_model_state[algo] = next_model_state[algo].detach()

    return prev_model_state

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

prev_model_state = {algo: None for algo, _ in specs.items()}
for batch_idx, (feature, is_first, is_last) in enumerate(dl):
    print(f"Processing Batch {batch_idx} {is_first} {is_last}")
    optimizer.zero_grad()
    prev_model_state = get_model_state(model, prev_model_state, is_first, feature)
    (predictions, losses, evaluations), next_model_state = model(feature, prev_model_state)
    prev_model_state = set_model_state(prev_model_state, is_last, next_model_state)
    flat_losses = []
    tree_map(lambda x: flat_losses.append(x), losses)
    total_loss = sum(flat_losses)

    print("Forwards Pass done!")
    print(f"Batch {batch_idx} loss: {total_loss}")
    total_loss.backward()
    optimizer.step()
    print("Backwards Pass done!")

    import ipdb; ipdb.set_trace()






