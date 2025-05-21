from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
import os
from typing import Any, Dict, List, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
import torch
from .utils import tree_map
from .trainer_utils import CustomRichProgressBar, normalize_state_dict, ModelCheckpointWithWandbSync
from .specs import CLRS30Algorithms, AlgorithmEnum, Spec
from .processors import ProcessorEnum
from .model import Model, ReconstMode
from .dataset import get_dataset

class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

@dataclass
class TrainerConfig:
    algos: Union[AlgorithmEnum, List[AlgorithmEnum]] = field(default_factory=lambda: CLRS30Algorithms)
    sizes: Optional[List[int]] = None
    num_steps: int = 10000
    static_batch_size: bool = True
    stacked: bool = False                                   # Paper found non-stacked training to be better      
    batch_size: int = 32
    seed: int = 42

    # Optimizer Settings
    learning_rate: float = 1e-3

    # Data Settings
    generate_on_the_fly: bool = True

    # Training Settings (Paper default values)
    encode_hints: bool = True                                
    decode_hints: bool = True                                
    use_lstm: bool = False                                   
    hint_reconst_mode: ReconstMode = ReconstMode.SOFT        
    hint_teacher_forcing: float = 0.0                        
    dropout: float = 0.0                                    

    # Model config
    hidden_dim: int = 128
    triplet_fts_size: int = 8
    use_ln: bool = True
    nb_heads: int = 1
    mp_steps: int = 1

    # Monitoring Settings
    val_check_interval: int = 500

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def __post_init__(self):
        if isinstance(self.algos, AlgorithmEnum):
            self.algos = [self.algos]


        self.algos = sorted(self.algos)

        self._specs = None
        if not self.stacked:
            # As per the generalise algorithmic learner paper
            # They train num_steps cycles
            self.num_steps = self.num_steps * len(self.algos)
            self.val_check_interval = min(self.val_check_interval * len(self.algos), 1500)
            self.test_check_interval = self.val_check_interval * 10

    @property
    def specs(self):
        if self._specs is None:
            raise ValueError("Specs not set, you must get the training dataloader first")
        return self._specs


    def get_dataloader(self, split: Split, num_workers: int = 0):

        if split == Split.TRAIN:
            seed = self.seed + 1
        elif split == Split.VAL:
            seed = self.seed + 2
        else:
            seed = self.seed + 3

        ds = get_dataset(self.algos, 
                         split=split.value,
                         sizes=self.sizes,
                         static_batch_size=self.static_batch_size,
                         generate_on_the_fly=self.generate_on_the_fly if split == Split.TRAIN else False,
                         stacked=self.stacked,
                         seed=seed)
        
        self._specs = ds.specs if split == Split.TRAIN else self._specs
        
        return ds.get_dataloader(batch_size=self.batch_size, 
                              shuffle=True if split == Split.TRAIN else False,
                              drop_last=True, # Avoid extra compilation 
                              num_workers=num_workers)
        
    
    def get_model(self):
        processors_kwargs = {
            "hidden_dim": self.hidden_dim, 
            "use_ln": self.use_ln, 
            "nb_heads": self.nb_heads, 
            "triplet_fts_size": self.triplet_fts_size,
            "mp_steps": self.mp_steps
        }
        if len(self.algos) == 1:
            processor = ProcessorEnum.triplet_gmpnn(**processors_kwargs)
        else:
            processor = ProcessorEnum.triplet_mpnn(**processors_kwargs)


        model = Model(specs=self.specs,
                      processor=processor,
                      hidden_dim=self.hidden_dim,
                      encode_hints=self.encode_hints,
                      decode_hints=self.decode_hints,
                      use_lstm=self.use_lstm,
                      hint_reconst_mode=self.hint_reconst_mode,
                      hint_teacher_forcing=self.hint_teacher_forcing,
                      dropout=self.dropout)
        
        return model
    
    def get_optimizer(self):
        return Adam(self.model.parameters(), 
                    lr=self.learning_rate, 
                    betas=(self.beta1, self.beta2), 
                    eps=self.eps, 
                    weight_decay=self.weight_decay)
    

class DataModule(pl.LightningDataModule):
    def __init__(self, config: TrainerConfig, num_workers: int = 0):
        super().__init__()
        self.config = config
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dl = self.config.get_dataloader(Split.TRAIN, num_workers=self.num_workers)
        self.val_dl = self.config.get_dataloader(Split.VAL, num_workers=self.num_workers)
        self.test_dl = self.config.get_dataloader(Split.TEST, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl
    

class TrainingModel(pl.LightningModule):
    def __init__(self, config: TrainerConfig, compile: bool = False):
        super().__init__()
        self.config = config
        self.compile = compile
        self.model = self.config.get_model()
        self.learning_rate = config.learning_rate
        self.save_hyperparameters(config.to_dict(), ignore=["model", "config", "compile"])
        self.examples_seen = defaultdict(int)
        self.steps_done = defaultdict(int)
        if self.compile:
            print("Compiling model using torch.compile...")
            self.model.compile()
        else:
            print("Model compilation disabled; skipping torch.compile.")

    def on_load_checkpoint(self, checkpoint):
        if self.model is None:
            self.configure_model()
        ckpt_state_dict = checkpoint["state_dict"]
        current_state_dict = self.state_dict()
        new_state_dict = normalize_state_dict(current_state_dict, ckpt_state_dict)
        checkpoint["state_dict"] = new_state_dict

    def log_metrics(self, evaluations, losses, phase: str):
        total_loss, scores = 0.0, []
        algo_metrics, total_metrics = {}, {}
        batch_size = self.config.batch_size

        for algo in evaluations.keys():
            if phase == "train":
                self.examples_seen[algo] += batch_size
                self.steps_done[algo] += 1
                # algo_metrics[f"{algo}/examples_seen"] = self.examples_seen[algo]
                # algo_metrics[f"{algo}/steps_done"] = self.steps_done[algo]
            flat_evals, flat_losses = [], []
            tree_map(lambda x: flat_evals.append(x), evaluations[algo])
            tree_map(lambda x: flat_losses.append(x), losses[algo])

            loss_algo = sum(flat_losses)
            total_loss = total_loss + loss_algo
            score_algo = (sum(flat_evals)/len(flat_evals)).detach().cpu().item()
            algo_metrics[f"{algo}/loss_{phase}"] = loss_algo.detach().cpu().item()
            algo_metrics[f"{algo}/score_{phase}"] = score_algo
            scores.append(score_algo)

        total_metrics[f"total/loss_{phase}"] = total_loss.detach().cpu().item()
        total_metrics[f"total/score_{phase}"] = sum(scores)/len(scores)

        if phase == "train":
            total_metrics[f"total/examples_seen"] = sum(self.examples_seen.values())

        on_step = True if phase == "train" else False
        on_epoch = True if phase != "train" else False
        self.log_dict(algo_metrics, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)
        self.log_dict(total_metrics, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        predictions, losses, evaluations = self.model(batch)
        total_loss = self.log_metrics(evaluations, losses, "train")
        return total_loss
    
    # def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
    #     step = self.global_step
    #     if step == 0 or step % self.config.test_check_interval == 0:
    #         print(f"Starting testing epoch...")
    #         self.trainer.test(datamodule=self.datamodule, ckpt_path=None, verbose=True)

    def validation_step(self, batch, batch_idx):
        prediction, losses, evaluations = self.model(batch)
        _ = self.log_metrics(evaluations, losses, "val")

    def test_step(self, batch, batch_idx):
        prediction, losses, evaluations = self.model(batch)
        _ = self.log_metrics(evaluations, losses, "test")

    def configure_optimizers(self):
        return Adam(self.model.parameters(), 
                    lr=self.learning_rate, 
                    betas=(0.9, 0.999), 
                    eps=1e-8, 
                    weight_decay=0.0)
    
    
def train(config: TrainerConfig, 
          run_name: str = 'run_1',
          project_name: str = 'clrs', 
          checkpoint_dir: str = './checkpoints',
          wandb_logging: bool = True,
          debug: bool = False,
          compile: bool = False) -> None:

    num_workers = 0 if debug else min(os.cpu_count() - 1, 8)
    project_name = project_name + "_debug" if debug else project_name
    
    pl.seed_everything(config.seed)

    # Dummy call to get the specs
    config.get_dataloader(Split.TRAIN, num_workers=0)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        id=run_name,
        version=run_name,
        log_model=False,
        save_dir=checkpoint_dir,
        reinit=True,
        mode="disabled" if not wandb_logging else "online",
        config=config.to_dict()
    )

    callbacks = [ CustomRichProgressBar()]
    callbacks.extend([
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="best",
                monitor='total/score_val',
                save_top_k=3,
                mode='max',
                auto_insert_metric_name=False,
                filename='best-step{step:07d}-Score:{total/score_val:.4f}-Loss:{total/loss_val:.4f}',
                wandb_verbose=False
            ),
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="backup",
                monitor='step',
                mode='max',
                save_top_k=2,
                every_n_train_steps=config.val_check_interval,
                auto_insert_metric_name=False,
                filename='last-step{step:07d}-Score:{total/score_val:.4f}-Loss:{total/loss_val:.4f}',
                wandb_verbose=False
            )
        ])

    trainer = pl.Trainer(
        default_root_dir=None,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,   
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        devices='auto',
        logger=wandb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=config.num_steps,
        limit_train_batches=None,
        limit_val_batches=None,
        check_val_every_n_epoch=None, # Turn off validation  per epoch
        val_check_interval=config.val_check_interval,
        enable_model_summary=True,
        # detect_anomaly=True
    )

    with trainer.init_module():
        model = TrainingModel(config, compile=compile)

    trainer.fit(model, datamodule=DataModule(config, num_workers=num_workers))




