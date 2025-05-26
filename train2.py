import time
import argparse
from contextlib import nullcontext
from collections import defaultdict

import torch
from torch.optim import Adam
from tqdm.auto import tqdm

from clrs.trainer import TrainerConfig, Split
from clrs.model import Model, ModelState
from clrs.utils import tree_flatten, tree_map


def make_optimizers(model: Model, lr_head: float, lr_shared: float):
    # pull out the single shared Processor instance
    shared = next(iter(model.models.values())).processor
    shared_params = list(shared.parameters())
    shared_opt    = Adam(shared_params, lr=lr_shared)

    shared_ids = {id(p) for p in shared_params}
    head_opts  = {}
    for algo, head in model.models.items():
        # only head-exclusive params
        head_params = [p for p in head.parameters() if id(p) not in shared_ids]
        head_opts[algo] = Adam(head_params, lr=lr_head)

    return head_opts, shared_opt

def to_device(obj, device):
    return tree_map(lambda x: x.to(device) if torch.is_tensor(x) else x, obj)


def train_epoch(dataloader, model, head_opts, shared_opt, train_state,
                device, n_steps, sync, use_streams):
    model.to(device)
    # zero gradients at start of epoch
    for opt in (*head_opts.values(), shared_opt):
        opt.zero_grad()

    step = 0
    t0 = time.time()
    data_iter = iter(dataloader)

    default_stream = torch.cuda.current_stream() if device.type=='cuda' else None

    pbar = tqdm(total=n_steps, desc="Training", unit="batch")
    while step < n_steps:
        try:
            features, is_first, is_last = next(data_iter)
        except StopIteration:
            break

        # Move to device
        features = to_device(features, device)

        # (re)initialize state for any algo that’s first in its cycle
        for algo, first in is_first.items():
            if first:
                train_state[algo] = model.init_model_state(algo, features[algo])

        # pick single‐batch algos from features keys
        algos = list(features.keys())

        if sync:
            # ───── SYNCHRONOUS ─────
            # optional per-head streams
            if use_streams and device.type=='cuda':
                streams = [torch.cuda.Stream() for _ in algos]
            else:
                streams = [default_stream]*len(algos)

            losses = []
            for algo, stream in zip(algos, streams):
                head = model.models[algo]
                head_opt = head_opts[algo]
                with torch.cuda.stream(stream) if stream else nullcontext(), \
                     torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                    (pred, loss_dict, evals), new_state = head(features[algo], train_state[algo])
                    loss_val = sum(tree_flatten(loss_dict))
                    loss_val.backward()
                    losses.append(loss_val)

                # update progress bar for this head
                pbar.set_postfix(algo=algo)
                # note: we don’t step optimizers until after all heads…

            if use_streams and device.type=='cuda':
                for s in streams:
                    s.synchronize()

            # step each head, then shared
            for algo in algos:
                head_opts[algo].step()
                head_opts[algo].zero_grad()

            shared_opt.step()
            shared_opt.zero_grad()

            step += 1
            pbar.update(1)

        else:
            # ───── ASYNCHRONOUS ─────
            if device.type=='cuda':
                streams = [torch.cuda.Stream() for _ in algos]
            else:
                streams = [None]*len(algos)
            events = []

            # enqueue forward+backward per head
            for algo, stream in zip(algos, streams):
                head = model.models[algo]
                head_opt = head_opts[algo]
                with torch.cuda.stream(stream) if stream else nullcontext(), \
                     torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
                    (pred, loss_dict, evals), new_state = head(features[algo], train_state[algo])
                    loss_val = sum(tree_flatten(loss_dict))
                    loss_val.backward()
                    evt = torch.cuda.Event() if stream else None
                    if evt: evt.record(stream)
                    events.append((algo, evt, head_opt, new_state, loss_val))

            # as soon as each backward is ready, step
            for algo, evt, head_opt, new_state, loss_val in events:
                if device.type=='cuda' and evt:
                    default_stream.wait_event(evt)

                # step head + shared
                head_opt.step(); head_opt.zero_grad()
                shared_opt.step(); shared_opt.zero_grad()


                # update and log
                step += 1
                pbar.update(1)
                pbar.set_postfix(algo=algo)

                if step >= n_steps:
                    break

        # end of one “batch” (sync or async)
    pbar.close()

    elapsed = time.time() - t0
    return step / elapsed

if __name__ == "__main__":
    mode = "async"
    use_streams = True
    stacked = True
    compile = True

    train_batches = 20 if stacked else 5

    if mode != "sync":
        train_batches *= 5 

    cfg = TrainerConfig(train_batches=train_batches, stacked=stacked)
    train_dl = cfg.get_dataloader(Split.TRAIN)
    model    = cfg.get_model(train_dl.dataset.specs)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_opts, shared_opt = make_optimizers(
        model,
        lr_head=cfg.learning_rate,
        lr_shared=cfg.learning_rate,
    )

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)
    train_state = {algo: None for algo in cfg.algorithms}
    n_steps     = cfg.num_train_steps


    bps = train_epoch(
        train_dl, model, head_opts, shared_opt,
        train_state, device, n_steps,
        sync=(mode=="sync"),
        use_streams=False
    )

    if cfg.stacked and mode == "sync":
        bps = bps * len(cfg.algorithms)

    print(f"\nMode: {mode}, UseStreams: {use_streams}, Stacked: {cfg.stacked} , Compile: {compile}, {bps} algo batches/sec")
