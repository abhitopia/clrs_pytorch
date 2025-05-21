import torch
from torch.optim import Adam
from collections import defaultdict
from pytorch_lightning import seed_everything
from clrs.processors.base import GraphFeatures
from clrs.processors.pgn import Reduction
from clrs.specs import AlgorithmEnum, Feature, Location, Stage, CLRS30Algorithms, Type
from clrs.processors import ProcessorEnum
from clrs.model import Model, ReconstMode, get_steps_mask, POS_INF, NEG_INF
from clrs.dataset_archive import get_dataset
from clrs.utils import batch_mask, expand
import clrs
import clrs.utils

def get_batches(algorithm: AlgorithmEnum, batch_size: int, size_small: int, size_large: int):
    ds1 = get_dataset(algos=algorithm,
                      sizes=[size_small],
                      num_samples=200,
                      stacked=False,
                      generate_on_the_fly=False,
                      string_matcher_override=False,
                      static_batch_size=False)
    ds2 = get_dataset(algos=algorithm,
                      sizes=[size_small, size_large],
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


def test_model_step(model: Model, b1: Feature, b2: Feature):

    t1, s1, n1 = b1[0], b1[1], b1[2]
    t2, s2, n2 = b2[0], b2[1], b2[2]

    spec = model.spec
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


        nfd1 = torch.cat([g1.node_fts, ps1, nps1], dim=-1)
        nfd2 = torch.cat([g2.node_fts, ps2, nps2], dim=-1)
        efd1 = g1.edge_fts 
        efd2 = g2.edge_fts 
        if nxe1 is not None:
            efd1 = torch.cat([g1.edge_fts, nxe1], dim=-1)
            efd2 = torch.cat([g2.edge_fts, nxe2], dim=-1)

        g1 = GraphFeatures(adj_mat=g1.adj_mat, 
                            node_fts=nfd1, 
                            edge_fts=efd1, 
                            graph_fts=g1.graph_fts)
        g2 = GraphFeatures(adj_mat=g2.adj_mat, 
                            node_fts=nfd2, 
                            edge_fts=efd2, 
                            graph_fts=g2.graph_fts)
        
        node_mask = expand(batch_mask(n2, NMax, 1), g2.node_fts) # [B, N]

        assert (g1.adj_mat == g2.adj_mat[:, :NMin, :NMin]).all()
        assert (g2.adj_mat[~edge_mask] == 0).all()
        assert (g1.node_fts == g2.node_fts[:, :NMin, :]).all()
        assert (g2.node_fts[~node_mask] == 0).all()
        assert (g2.edge_fts[~edge_mask] == 0).all()
        assert (g1.edge_fts == g2.edge_fts[:, :NMin, :NMin, :]).all()
        assert (g1.graph_fts == g2.graph_fts).all()

        pred1, raw_pred1 = model.decoder(g1, num_nodes=None)
        pred2, raw_pred2 = model.decoder(g2, num_nodes=n2)

        for stage in [Stage.OUTPUT, Stage.HINT]:
            raw_out1 = raw_pred1[stage]
            raw_out2 = raw_pred2[stage]

            for key in raw_out1.keys():
                _, location, type_, _  = spec[key]
                fill_value = 0.0 if type_ == Type.SCALAR else NEG_INF
                v1 = raw_out1[key]
                v2 = raw_out2[key]
                try:
                    offset = 1 if type_ == Type.CATEGORICAL else 0
                    num_node_dims = v1.ndim - offset - 1
                    if num_node_dims > 0:
                        mask = expand(batch_mask(n2, NMax, num_node_dims), v2)
                        assert (v2[~mask] == fill_value).all()
                        slice_tuple = [slice(None, None)] + [slice(None, NMin)] * (num_node_dims) + [slice(None, None)] * (offset)
                        assert (v1 == v2[slice_tuple]).all()
                    else:
                        assert (v1 == v2).all()
                except Exception as e:
                    print(f"Failed for key: {key} in stage: {stage} for type: {type_} and location: {location}")
                    import ipdb; ipdb.set_trace()
                    raise e
                
        for stage in [Stage.OUTPUT, Stage.HINT]:
            pred_out1 = pred1[stage]
            pred_out2 = pred2[stage]
            for key in pred_out1.keys():
                _, location, type_, _  = spec[key]
                v1 = pred_out1[key]
                v2 = pred_out2[key]
                try:
                    offset = 1 if type_ == Type.CATEGORICAL else 0
                    num_node_dims = v1.ndim - offset - 1
                    if num_node_dims > 0:
                        mask = expand(batch_mask(n2, NMax, num_node_dims), v2)
                        assert (v2[~mask] == 0.0).all()
                        slice_tuple = [slice(None, None)] + [slice(None, NMin)] * (num_node_dims) + [slice(None, None)] * (offset)
                        assert (v1 == v2[slice_tuple]).all()
                    else:
                        assert (v1 == v2).all()
                except Exception as e:
                    print(f"Failed for key: {key} in stage: {stage} for type: {type_} and location: {location}")
                    import ipdb; ipdb.set_trace()
                    raise e


def compare_values_step(spec, key, v1, v2, n2, NMin, NMax):
    stage, location, type_, _  = spec[key]
        
    try:
        offset = 1 if type_ == Type.CATEGORICAL else 0
        prior_dims = 1
        num_node_dims = v1.ndim - offset - prior_dims
        if num_node_dims > 0:
            mask = expand(batch_mask(n2, NMax, num_node_dims), v2, prior_dims=prior_dims-1)
            assert (v2[~mask] == 0.0).all()
            slice_tuple = [slice(None, None)]*prior_dims + [slice(None, NMin)] * (num_node_dims) + [slice(None, None)] * (offset)
            assert torch.allclose(v1, v2[slice_tuple])
        else:
            assert torch.allclose(v1, v2)
                
    except Exception as e:
        print(f"Failed for key: {key} in stage: {stage} for type: {type_} and location: {location}")
        import ipdb; ipdb.set_trace()
        raise e


def test_model_output(model: Model, b1: Feature, b2: Feature):
    clrs.model.set_use_num_nodes(False)
    p1, l1, e1 = model(b1)
    clrs.model.set_use_num_nodes(True)
    p1nn, l1nn, e1nn = model(b1)
    p2, l2, e2 = model(b2)

    _, s1, n1 = b1[0], b1[1], b1[2]
    t2, s2, n2 = b2[0], b2[1], b2[2]
    NMin = max(n1).item()
    NMax = t2['input']['pos'].size(1)

    spec = model.spec

    for stage in [Stage.OUTPUT, Stage.HINT]:
        for key in p1[stage].keys():
            if stage == Stage.OUTPUT:
                compare_values_step(spec, key, p1[stage][key], p2[stage][key], n2, NMin, NMax)
            else:
                h1_key, h2_key = {key: p1[stage][key]}, {key: p2[stage][key]}
                Smin, Smax = h1_key[key].size(0), h2_key[key].size(0)
                for step in range(Smax):
                    v2 = model.get_hint_at_step(h2_key, step)[key]
                    if step < Smin: # We don't care about the padding steps
                        v1 = model.get_hint_at_step(h1_key, step)[key]
                        compare_values_step(spec, key, v1, v2, n2, NMin, NMax)


            try:
                if key == 'pred_mask' and stage == Stage.OUTPUT and 'pred' in spec and  spec['pred'][2] == Type.PERMUTATION_POINTER:
                    # Skipping mask_1 for permutation pointer
                    continue
                try:
                    assert (e1[stage][key] == e1nn[stage][key]).all()
                    assert (l1[stage][key] == l1nn[stage][key]).all()
                    assert (e1[stage][key] == e2[stage][key]).all()
                    assert (l1[stage][key] == l2[stage][key]).all()
                except Exception as e:
                    print(f"Testing all close for: {key} in stage: {stage} of type: {spec[key][2]} and location: {spec[key][1]}")
                    assert torch.allclose(l1[stage][key], l2[stage][key])
                    assert torch.allclose(e1[stage][key], e2[stage][key])
                    assert torch.allclose(l1[stage][key], l1nn[stage][key])
                    assert torch.allclose(e1[stage][key], e1nn[stage][key])
            except Exception as e:
                print(f"Failed evaluation for key: {key} in stage: {stage}")
                import ipdb; ipdb.set_trace()
                raise e


def test_static_batch(algorithm: AlgorithmEnum, processor: ProcessorEnum, size_small: int, size_large: int):
    torch.set_printoptions(profile="full", precision=16)
    clrs.utils.set_bias_value(1.0)
    hidden_dim = 128
    batch_size = 32
    encode_hints = True
    decode_hints = True
    use_lstm = False
    hint_reconst_mode = ReconstMode.SOFT
    hint_teacher_forcing = 1.0
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
    
    # model.compile()
    
    model.eval()  # This is important to prevent noise from being injected in log_sinkhorn
    spec = specs[algorithm]

    test_model_step(model, b1[algorithm], b2[algorithm])
    test_model_output(model, b1[algorithm], b2[algorithm])

    clrs.utils.set_bias_value(0.0)


def test_gradient_step(algorithm: AlgorithmEnum, processor: ProcessorEnum, size_small: int, size_large: int):
    hidden_dim = 128
    batch_size = 32
    encode_hints = True
    decode_hints = True
    use_lstm = False
    hint_reconst_mode = ReconstMode.SOFT
    hint_teacher_forcing = 1.0
    dropout = 0.0
    reduction = Reduction.MAX


    ds = get_dataset(algos=algorithm,
                      sizes=[size_small, size_large],
                      num_samples=200,
                      stacked=False,
                      generate_on_the_fly=False,
                      string_matcher_override=False,
                      static_batch_size=True)
    
    dl = ds.get_dataloader(
        batch_size=batch_size,
        shuffle=True, 
        drop_last=False,
        num_workers=0
    )

    processor = processor(hidden_dim=hidden_dim, mp_steps=1, reduction=reduction)
    model = Model(specs=ds.specs,
                processor=processor,
                hidden_dim=hidden_dim,
                encode_hints=encode_hints,
                decode_hints=decode_hints,
                use_lstm=use_lstm,
                hint_reconst_mode=hint_reconst_mode,
                hint_teacher_forcing=hint_teacher_forcing,
                dropout=dropout
                ).models[algorithm]
    
    optimizer = Adam(model.parameters(), lr=1e-3)

    for idx, batch in enumerate(dl):
        optimizer.zero_grad()
        predictions, losses, evaluations = model(batch[algorithm])
        # import ipdb; ipdb.set_trace()
        total_loss = 0.0
        for stage in [Stage.OUTPUT, Stage.HINT]:
            for key in losses[stage].keys():
                total_loss += losses[stage][key]
        # total_loss += losses[Stage.OUTPUT]['pi'] 
        # total_loss += losses[Stage.HINT]['pi_h']
        # total_loss += losses[Stage.HINT]['reach_h']
        print(f"Total loss: {total_loss}")
        with torch.autograd.detect_anomaly():
            total_loss.backward()
            optimizer.step()
        # import ipdb; ipdb.set_trace()
        if idx > 3:
            break



if __name__ == "__main__":
    seed_everything(42)
    algorithms = CLRS30Algorithms
    # algorithms = [AlgorithmEnum.naive_string_matcher]
    # algorithms = [AlgorithmEnum.matrix_chain_order]
    # algorithms = [AlgorithmEnum.lcs_length]
    # algorithms = [AlgorithmEnum.bfs]
    # algorithms = [AlgorithmEnum.bridges]
    # algorithms = [AlgorithmEnum.insertion_sort]
    processors = list(ProcessorEnum)
    processors = [ProcessorEnum.triplet_gmpnn]

    for processor in processors:
        for algorithm in algorithms:
            print(f"Testing {algorithm.name} with {processor.name}")
            # test_static_batch(algorithm, 
            #                 processor, 
            #                 size_small=4, 
            #                 size_large=12)
            test_gradient_step(algorithm, 
                            processor, 
                            size_small=4, 
                            size_large=12)
        print(f"Testing {algorithm.name} passed")