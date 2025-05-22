#! python
import os
import torch
from pathlib import Path
from typing import List, Optional
import typer
from clrs.specs import Algorithm, CLRS30Algorithms
from clrs.trainer import TrainerConfig, train
import logging
from rich import print

# To prevent pesky inductor warnings
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(__file__).parent / "torch_compile_cache")

def cache_dir() -> Path:
    current_file = Path(__file__)
    if 'this_studio' in str(current_file) and 'teamspace' in str(current_file):
        # Lightning Studio
        return Path("/teamspace/uploads/clrs_cache")
    else:
        # Local
        return Path("/tmp/clrs_cache")

app = typer.Typer(
    name="clrs",
    help="CLRS Training CLI",
    add_completion=False,
    pretty_exceptions_show_locals=False
)

@app.command("train")
def main(
    algos: List[Algorithm] = typer.Option(CLRS30Algorithms, "--algos", "-a", help="Algorithms to train", ),
    sizes: List[int] = typer.Option([4, 7, 11, 13, 16], "--sizes", "-s", help="Sizes to train, max value is used for validation"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    num_train_batches: int = typer.Option(10000, "--num-train-batches", "-t", help="Number of training batches per algorithm"),
    num_val_batches: int = typer.Option(10, "--num-val-batches", "-v", help="Number of validation batches per algorithm"),
    chunk_size: int = typer.Option(16, "--chunk-size", "--cs", help="The maximum number of hint steps per batch, <0 means no chunking"),
    run_name: str = typer.Option("run", "--run-name", "-r", help="Run name"),
    project_name: str = typer.Option("clrs", "--project-name", "-p", help="Project name"),
    ckpt_dir: Path = typer.Option("./checkpoints", "--ckpt-dir", help="Checkpoint directory"),
    seed: int = typer.Option(42, "--seed", help="Seed"),
    dynamic_batch_size: bool = typer.Option(False, "--dynamic-batch-size", "-DB", help="Use non-static batch size", is_flag=True, flag_value=True),
    stacked: bool = typer.Option(False, "--stacked", "-S", help="Stacked training", is_flag=True, flag_value=True),
    compile: bool = typer.Option(False, "--compile", "-C", help="Compile model", is_flag=True, flag_value=True),
    debug: bool = typer.Option(False, "--debug", "-D", help="Debug mode", is_flag=True, flag_value=True),
    cache_dir: Path = typer.Option(cache_dir(), "--cache-dir", help="Cache directory"),
) -> None:
    
    if stacked:
        assert len(algos) > 1, "Stacked training requires at least two algorithms"
        
    config = TrainerConfig(
        algorithms=algos,
        sizes=sizes,
        seed=seed,
        stacked=stacked,
        batch_size=batch_size,
        static_batch_size=not dynamic_batch_size,
        chunk_size=chunk_size,
        cache_dir=cache_dir,
        train_batches=num_train_batches,
        val_batches=num_val_batches,
    )

    print("Config:", config.to_dict())
    train(
        config=config,
        project_name=project_name,
        run_name=run_name,
        checkpoint_dir=ckpt_dir,
        compile=compile,
        debug=debug
    )

if __name__ == "__main__":
    app()