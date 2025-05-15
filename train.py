#! python
import os
from pathlib import Path
import typer
from clrs.specs import Algorithms, CLRS30Algorithms
from clrs.trainer import TrainerConfig, train
import logging

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
    algos: list[Algorithms] = typer.Option(CLRS30Algorithms, "--algos", "-a", help="Algorithms to train", ),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    run_name: str = typer.Option("run_1", "--run-name", "-n", help="Run name"),
    project_name: str = typer.Option("clrs", "--project-name", "-p", help="Project name"),
    ckpt_dir: Path = typer.Option("./checkpoints", "--ckpt-dir", "-c", help="Checkpoint directory"),
    static_num_hints: bool = typer.Option(False, "--static-num-hints", "-s", help="Use static number of hints", is_flag=True, flag_value=True),
    seed: int = typer.Option(42, "--seed", "-s", help="Seed"),
    compile: bool = typer.Option(False, "--compile"),
) -> None:
    
    train(
        config=TrainerConfig(algos=algos, 
                             seed=seed, 
                             batch_size=batch_size, 
                             uniform_hint_steps=static_num_hints),
        project_name=project_name,
        run_name=run_name,
        checkpoint_dir=ckpt_dir,
        compile=compile
    )

if __name__ == "__main__":
    app()