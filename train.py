#! python
import os
import torch
from pathlib import Path
from typing import List
import typer
from clrs.specs import AlgorithmEnum, CLRS30Algorithms
from clrs.trainer import TrainerConfig, train
import logging
from rich import print

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
    algos: List[AlgorithmEnum] = typer.Option(CLRS30Algorithms, "--algos", "-a", help="Algorithms to train", ),
    train_sizes: List[int] = typer.Option([4, 7, 11, 13, 16], "--train-sizes", "-s", help="Sizes to train"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    run_name: str = typer.Option("run_1", "--run-name", "-n", help="Run name"),
    project_name: str = typer.Option("clrs", "--project-name", "-p", help="Project name"),
    ckpt_dir: Path = typer.Option("./checkpoints", "--ckpt-dir", "-c", help="Checkpoint directory"),
    static_num_hints: bool = typer.Option(False, "--static-num-hints", "--static", help="Use static number of hints", is_flag=True, flag_value=True),
    seed: int = typer.Option(42, "--seed", "-sd", help="Seed"),
    stacked: bool = typer.Option(False, "--stacked", "-S", help="Stacked training", is_flag=True, flag_value=True),
    compile: bool = typer.Option(False, "--compile"),
    debug: bool = typer.Option(False, "--debug", "-D", help="Debug mode", is_flag=True, flag_value=True),
) -> None:
    
    if stacked:
        assert len(algos) > 1, "Stacked training requires at least two algorithms"
        
    config = TrainerConfig(algos=algos, 
                        seed=seed, 
                        stacked=stacked,
                        batch_size=batch_size, 
                        uniform_hint_steps=static_num_hints)
    config.train_data["sizes"] = train_sizes

   
    if torch.cuda.is_available() and compile:
        print("WARNING: Currently compiling seems to be broken on CUDA. Use at your own risk!")

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