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
import warnings

import torch._logging as torch_logging

# Control the logging to help debug compilation issues
torch_logging.set_logs(
#     # dynamo=logging.DEBUG,
#     # inductor=logging.DEBUG,
    recompiles=True,
#     guards=True,
    graph_breaks=True
)

warnings.filterwarnings("ignore", module=r"torch\._inductor(\.|$)")
warnings.filterwarnings("ignore", module=r"torch\.utils\._sympy\.")

# To prevent pesky inductor warnings
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(__file__).parent / "torch_compile_cache")


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
    num_val_batches: int = typer.Option(50, "--num-val-batches", "-v", help="Number of validation batches per algorithm"),
    chunk_size: int = typer.Option(16, "--chunk-size", "--cs", help="The maximum number of hint steps per batch, <0 means no chunking"),
    run_name: str = typer.Option("run", "--run-name", "-r", help="Run name"),
    project_name: str = typer.Option("clrs", "--project-name", "-p", help="Project name"),
    ckpt_dir: Path = typer.Option("./checkpoints", "--ckpt-dir", help="Checkpoint directory"),
    seed: int = typer.Option(42, "--seed", help="Seed"),
    dynamic_batch_size: bool = typer.Option(False, "--dynamic-batch-size", "-DB", help="Use non-static batch size", is_flag=True, flag_value=True),
    stacked: bool = typer.Option(False, "--stacked", "-S", help="Stacked training", is_flag=True, flag_value=True),
    no_static_hints_as_input: bool = typer.Option(False, "--no-static-hints-as-input", "-NSHAI", help="Don't move static hints to input", is_flag=True, flag_value=True),
    sorting_output_as_permutation: bool = typer.Option(False, "--sorting-output-as-permutation", "-SOAP", help="Sorting output as permutation", is_flag=True, flag_value=True),
    no_random_pos_embedding: bool = typer.Option(False, "--no-random-pos-embedding", "-NRPE", help="Don't use Randomize position embedding", is_flag=True, flag_value=True),
    compile: bool = typer.Option(False, "--compile", "-C", help="Compile model", is_flag=True, flag_value=True),
    debug: bool = typer.Option(False, "--debug", "-D", help="Debug mode", is_flag=True, flag_value=True),
    val_check_interval: int = typer.Option(1000, "--val-check-interval", "-vci", help="Validation check interval"),
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
        train_batches=num_train_batches,
        val_batches=num_val_batches,
        random_pos_embedding=not no_random_pos_embedding,
        static_hints_as_input=not no_static_hints_as_input,
        sorting_output_as_permutation=sorting_output_as_permutation,
    )

    print("Config:", config.to_dict())
    train(
        config=config,
        project_name=project_name,
        run_name=run_name,
        val_check_interval=val_check_interval,
        checkpoint_dir=ckpt_dir,
        compile=compile,
        debug=debug
    )

if __name__ == "__main__":
    app()