import torch
from collections import defaultdict
from pytorch_lightning import seed_everything
from clrs.processors.base import GraphFeatures
from clrs.processors.pgn import Reduction
from clrs.specs import AlgorithmEnum, Feature, Stage, CLRS30Algorithms
from clrs.processors import ProcessorEnum
from clrs.model import Model, ReconstMode
from clrs.dataset import get_dataset
from clrs.utils import batch_mask, expand
import clrs.utils

def get_batches(algorithm: AlgorithmEnum, batch_size: int, size_small: int, size_large: int):
    ds1 = get_dataset(algos=algorithm,
                      trajectory_sizes=[size_small],
                      num_samples=200,
                      stacked=False,
                      generate_on_the_fly=False,
                      string_matcher_override=False,
                      static_batch_size=False)
    ds2 = get_dataset(algos=algorithm,
                      trajectory_sizes=[size_small, size_large],
                      num_samples=200,
                      stacked=False,
                      generate_on_the_fly=False,
                      string_matcher_override=False,
                      static_batch_size=True)

    dl1 = ds1.get_dataloader(
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )
    dl2 = ds2.get_dataloader(
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )

    b1, b2 = next(iter(dl1)), next(iter(dl2))
    return b1, b2, ds1.specs


def test_encoding(model: Model, b1: Feature, b2: Feature):

    # saved = defaultdict(list)

    # # 2) define a hook fn that saves each module's output
    # def save_hook(mod, inp, out):
    #     # mod.__class__.__name__ is e.g. "Linear" or "LayerNorm"
    #     saved[mod].append(out.detach().clone())

    # handles = []
    # for mod in [model.processor.msgs_dummy,
    #             model.processor.fc_out_source, 
    #             model.processor.fc_out_messages, 
    #             model.processor.layer_norm]:
    #     handles.append(mod.register_forward_hook(save_hook))

    t1, s1, n1 = b1[0], b1[1], b1[2]
    t2, s2, n2 = b2[0], b2[1], b2[2]

    # global _BIAS_VALUE
    # _BIAS_VALUE = 1.0

    i1, h1, o1 = t1[Stage.INPUT], t1[Stage.HINT], t1[Stage.OUTPUT]
    i2, h2, o2 = t2[Stage.INPUT], t2[Stage.HINT], t2[Stage.OUTPUT]

    num_steps = max(s1).item()

    NMin = max(n1).item()
    NMax = i2['pos'].size(1)
    batch_size = i1['pos'].size(0)
    hidden_dim = model.hidden_dim

    ps1 = torch.zeros((batch_size, NMin, hidden_dim))
    ps2 = torch.zeros((batch_size, NMax, hidden_dim))

    for step in range(num_steps):
        h1_step = model.get_hint_at_step(h1, step)
        h2_step = model.get_hint_at_step(h2, step)

        # allowed_h_keys = ['best_high', 'best_low', 'best_sum', 'i', 'j', 'sum']
        # allowed_h_keys = []
        # allowed_i_keys = ['pos', 'pred_h']

        # import ipdb; ipdb.set_trace()
        # h1_step = {k: v for k, v in h1_step.items() if k in allowed_h_keys}
        # h2_step = {k: v for k, v in h2_step.items() if k in allowed_h_keys}
        # i1 = {k: v for k, v in i1.items() if k in allowed_i_keys}
        # i2 = {k: v for k, v in i2.items() if k in allowed_i_keys}

        g1 = model.encoder(i1, h1_step, None)
        g2 = model.encoder(i2, h2_step, n2)

        node_mask = expand(batch_mask(n2, NMax, 1), g2.node_fts) # [B, N]
        edge_mask = batch_mask(n2, NMax, 2)

        assert (g1.adj_mat == g2.adj_mat[:, :NMin, :NMin]).all()
        assert (g2.adj_mat[~edge_mask] == 0).all()
        assert (g1.node_fts == g2.node_fts[:, :NMin, :]).all()
        assert (g2.node_fts[~node_mask] == 0).all()
        assert (g2.edge_fts[~edge_mask] == 0).all()
        assert (g1.edge_fts == g2.edge_fts[:, :NMin, :NMin, :]).all()
        assert (g1.graph_fts == g2.graph_fts).all()

        nps1, nxe1 = model.processor(g1, processor_state=ps1, num_nodes=None)
        nps2, nxe2 = model.processor(g2, processor_state=ps2, num_nodes=n2)

        if nxe1 is None:
            assert nxe2 is None
        else:
            assert (nxe2[~(expand(edge_mask, nxe2))] == 0).all()
            assert (nxe1 == nxe2[:, :NMin, :NMin, :]).all()
        
        assert (nps2[~node_mask] == 0).all()
        assert (nps1 == nps2[:, :NMin, :]).all()


def test_static_batch(algorithm: AlgorithmEnum, processor: ProcessorEnum, size_small: int, size_large: int):
    clrs.utils.set_bias_value(1.0)
    hidden_dim = 128
    batch_size = 32
    encode_hints = True
    decode_hints = True
    use_lstm = False
    hint_reconst_mode = ReconstMode.SOFT
    hint_teacher_forcing = 0.0
    dropout = 0.0
    reduction = Reduction.MAX

    b1, b2, specs = get_batches(algorithm, batch_size, size_small, size_large)
    processor = processor(hidden_dim=hidden_dim, mp_steps=1, reduction=reduction)
    model = Model(specs=specs,
                processor=processor,
                hidden_dim=hidden_dim,
                encode_hints=encode_hints,
                decode_hints=decode_hints,
                use_lstm=use_lstm,
                hint_reconst_mode=hint_reconst_mode,
                hint_teacher_forcing=hint_teacher_forcing,
                dropout=dropout
                ).models[algorithm]
    spec = specs[algorithm]

    test_encoding(model, b1[algorithm], b2[algorithm])

    clrs.utils.set_bias_value(0.0)



if __name__ == "__main__":
    seed_everything(42)
    algorithms = CLRS30Algorithms
    # algorithms = [AlgorithmEnum.naive_string_matcher]
    # algorithms = [AlgorithmEnum.matrix_chain_order]
    # algorithms = [AlgorithmEnum.optimal_bst]
    processors = list(ProcessorEnum)
    processors = [ProcessorEnum.pgn]

    for processor in processors:
        for algorithm in algorithms:
            print(f"Testing {algorithm.name} with {processor.name}")
            test_static_batch(algorithm, 
                            processor, 
                            size_small=4, 
                            size_large=12)
        print(f"Testing {algorithm.name} passed")