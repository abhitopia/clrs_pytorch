from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import os
from typing import Dict, List, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
import torch
from .utils import tree_flatten
from .trainer_utils import CustomRichProgressBar, ModelCheckpointWithWandbSync
from .specs import CLRS30Algorithms, Algorithm, Spec, Feature, Stage
from .processors import Processor
from .model import Model, ModelState, ReconstMode
from .dataset import AlgoFeatureDataset, DictFeatureBatch, StackedAlgoFeatureDataset, CyclicAlgoFeatureDataset

class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"

@dataclass
class TrainerConfig:
    algorithms: Union[Algorithm, List[Algorithm]] = field(default_factory=lambda: CLRS30Algorithms)
    sizes: List[int] = field(default_factory=lambda: [4, 7, 11, 13, 16])  # Training sizes, max size is used for validation
    train_batches: int = 10000                              # Number of total training batches per algorithm (Same as paper)
    val_batches: int = 10                                   # Number of validation batches per algorithm
    static_batch_size: bool = True                          # If True, then the num nodes and num steps are fixed for each batch per algorithm
    stacked: bool = False                                   # Paper found non-stacked training to be better      
    chunk_size: Optional[int] = 16                          # Number of hints per batch, if <0 or None, then no chunking 
    batch_size: int = 32                                    # Number of samples per batch
    seed: int = 42                                          # Random seed used for data generation

    # Argorithm Kwargs
    random_pos_embedding: bool = True                      # Randomize position embedding (Paper default is True)
    static_hints_as_input: bool = True                     # Move static hints to input (Paper default is True)
    sorting_output_as_permutation: bool = False            # Sorting output as permutation (Paper default is True but it is extremely slow because of log_sinkhorn)

    # Optimizer Settings
    learning_rate: float = 1e-3

    # Training Settings (Paper default values)
    encode_hints: bool = True                                # Encode hints into processor embeddings per hint step
    decode_hints: bool = True                                # Decode hints from processor embeddings per hint step
    use_lstm: bool = False                                   # Use LSTM across hint steps
    hint_reconst_mode: ReconstMode = ReconstMode.SOFT        # Reconstruction mode for hints
    hint_teacher_forcing: float = 0.0                        # Teacher forcing ratio for hints, 0.0 means no teacher forcing
    dropout: float = 0.0                                     # Dropout probability

    # Model config
    hidden_dim: int = 128                                    # Hidden dimension of the processor and everywhere else
    triplet_fts_size: int = 8                                # Number of features for the triplet GNN (if using triplet variant of GNN)
    use_ln: bool = True                                      # Use layer normalization
    nb_heads: int = 1                                        # Number of attention heads (if using GAT variant of GNN)
    mp_steps: int = 1                                        # Number of message passing steps of GNN per hint step
    skip_scalar_eval: bool = True                            # Skip scalar evaluation

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def __post_init__(self):
        if isinstance(self.algorithms, Algorithm):
            self.algorithms = [self.algorithms]

        if self.chunk_size is not None and self.chunk_size < 0:
            self.chunk_size = None

        self.algorithms = sorted(self.algorithms)
        if not self.stacked:
            # As per the generalise algorithmic learner paper, they train num_steps cycles
            self.num_train_steps = self.train_batches * len(self.algorithms)
        else:
            self.num_train_steps = self.train_batches

    def get_dataloader(self, split: Split, num_workers: int = 0):
        seed = self.seed + (1 if split == Split.VAL else 0)
        num_batches = self.train_batches if split == Split.TRAIN else self.val_batches
        algo_datasets = []

        algo_kwargs = {
            "random_pos_embedding": self.random_pos_embedding,
            "static_hints_as_input": self.static_hints_as_input,
            "sorting_output_as_permutation": self.sorting_output_as_permutation
        }

        for algorithm in self.algorithms:
            algo_sizes = self.sizes

            if split == "val":
                algo_sizes = [max(algo_sizes)]

            # As per the generalise algorithmic learner paper, we replace the max length with 5/4 of the max length for 
            # string matching algorithms for training. For validation, we use the max length.
            if algorithm in [Algorithm.naive_string_matcher, Algorithm.kmp_matcher] and split == Split.TRAIN:
                max_length = max(algo_sizes)
                max_length = (max_length * 5) // 4
                algo_sizes = [max_length]*len(algo_sizes)

            algo_datasets.append(AlgoFeatureDataset(
                                                algorithm=algorithm,
                                                sizes=algo_sizes,
                                                chunk_size=self.chunk_size,
                                                batch_size=self.batch_size,
                                                num_batches=num_batches,
                                                seed=seed,
                                                static_batch_size=self.static_batch_size,
                                                algo_kwargs=algo_kwargs))    
            
        dataset = StackedAlgoFeatureDataset(algo_datasets) if self.stacked else CyclicAlgoFeatureDataset(algo_datasets)
        return dataset.get_dataloader(num_workers=num_workers)
        
    
    def get_model(self, specs: Dict[Algorithm, Spec]):
        processors_kwargs = {
            "hidden_dim": self.hidden_dim, 
            "use_ln": self.use_ln, 
            "nb_heads": self.nb_heads, 
            "triplet_fts_size": self.triplet_fts_size,
            "mp_steps": self.mp_steps
        }
        if len(self.algorithms) == 1:
            # Paper uses triplet GMPNN for single algorithm training
            processor = Processor.triplet_gmpnn(**processors_kwargs)
        else:
            # Paper uses triplet MPNN for multi-algorithm training
            processor = Processor.triplet_mpnn(**processors_kwargs)

        model = Model(specs=specs,
                      processor=processor,
                      hidden_dim=self.hidden_dim,
                      encode_hints=self.encode_hints,
                      decode_hints=self.decode_hints,
                      use_lstm=self.use_lstm,
                      hint_reconst_mode=self.hint_reconst_mode,
                      hint_teacher_forcing=self.hint_teacher_forcing,
                      skip_scalar_eval=self.skip_scalar_eval,
                      dropout=self.dropout)
        
        return model
    
    def get_optimizer(self):
        return Adam(self.model.parameters(), 
                    lr=self.learning_rate, 
                    betas=(self.beta1, self.beta2), 
                    eps=self.eps, 
                    weight_decay=self.weight_decay)
    

class TrainingModel(pl.LightningModule):
    def __init__(self, model: Model, config: TrainerConfig):
        super().__init__()
        self.config = config
        self.model = model
        self.learning_rate = config.learning_rate
        self.save_hyperparameters(config.to_dict(), ignore=["model", "config", "compile", "specs"])
        self.examples_seen = defaultdict(int)
        self.batches_seen = defaultdict(int)
        self.train_model_state = {algo: None for algo in self.model.specs.keys()}
        self.val_model_state = {algo: None for algo in self.model.specs.keys()}

    def log_metrics(self, split: Split, evaluations, losses, is_first: Dict[Algorithm, bool], is_last: Dict[Algorithm, bool]):
        total_loss, total_output_scores, total_hint_scores = 0.0, 0.0, 0.0
        total_len_loss, total_len_output_score, total_len_hint_score = 0, 0, 0
        algo_metrics, total_metrics = {}, {}
        batch_size = self.config.batch_size

        for algo in evaluations.keys():
            if split == Split.TRAIN:
                self.examples_seen[algo] += batch_size * (1 if is_first[algo] else 0)
                self.batches_seen[algo] += 1
                algo_metrics[f"{algo}/examples_seen"] = self.examples_seen[algo]
                algo_metrics[f"{algo}/batches_seen"] = self.batches_seen[algo]

            flat_hints_eval = tree_flatten(evaluations[algo][Stage.HINT])
            flat_hints_loss = tree_flatten(losses[algo][Stage.HINT])

            # Loss
            loss_algo = sum(flat_hints_loss) 
            len_loss_algo = len(flat_hints_loss)

            # Hint Score
            hint_score_algo = sum(flat_hints_eval).detach().cpu().item()
            len_hint_score_algo = len(flat_hints_eval)
            algo_metrics[f"{algo}/hint_score_{split}"] = (hint_score_algo / len_hint_score_algo)


            # Output score
            output_score_algo = 0.0
            len_output_score_algo = 0

            if is_last[algo]:
                flat_outputs_eval = tree_flatten(evaluations[algo][Stage.OUTPUT])
                flat_outputs_loss = tree_flatten(losses[algo][Stage.OUTPUT])

                # Loss 
                loss_algo = loss_algo + sum(flat_outputs_loss)
                len_loss_algo = len_loss_algo + len(flat_outputs_loss)

                # Output Score
                output_score_algo = output_score_algo + sum(flat_outputs_eval).detach().cpu().item()
                len_output_score_algo = len_output_score_algo + len(flat_outputs_eval)
                algo_metrics[f"{algo}/output_score_{split}"] = (output_score_algo / len_output_score_algo)

            # Loss
            total_loss = total_loss + loss_algo
            total_len_loss = total_len_loss + len_loss_algo

            # Total Scores Output and Hint
            total_hint_scores = total_hint_scores + hint_score_algo
            total_output_scores = total_output_scores + output_score_algo
            total_len_hint_score = total_len_hint_score + len_hint_score_algo
            total_len_output_score = total_len_output_score + len_output_score_algo

            algo_metrics[f"{algo}/loss_{split}"] = (loss_algo / len_loss_algo).detach().cpu().item()


        loss = total_loss / total_len_loss
        total_metrics[f"total/loss_{split}"] = loss.detach().cpu().item()
        if total_len_hint_score > 0:
            total_metrics[f"total/hint_score_{split}"] = total_hint_scores / total_len_hint_score

        if total_len_output_score > 0:
            total_metrics[f"total/output_score_{split}"] = total_output_scores / total_len_output_score

        if split == Split.TRAIN:
            total_metrics[f"total/examples_seen"] = sum(self.examples_seen.values())

        on_step = True if split == Split.TRAIN else False
        on_epoch = True if split != Split.TRAIN else False
        self.log_dict(algo_metrics, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size)
        self.log_dict(total_metrics, on_step=on_step, on_epoch=on_epoch, batch_size=batch_size, prog_bar=True)
        return total_loss
    
    def get_model_state(self, split: Split, is_first: Dict[Algorithm, bool], features: Dict[Algorithm, Feature]):
        new_model_state = {}
        prev_model_state = self.train_model_state if split == Split.TRAIN else self.val_model_state
        for algo, batch_is_first in is_first.items():
            if batch_is_first:
                assert prev_model_state[algo] is None
                new_model_state[algo] = self.model.init_model_state(algo, features[algo])
            else:
                new_model_state[algo] = prev_model_state[algo]
        return new_model_state
    
    def set_model_state(self, split: Split, is_last: Dict[Algorithm, bool], model_state: Dict[Algorithm, ModelState]):
        prev_model_state = self.train_model_state if split == Split.TRAIN else self.val_model_state
        for algo, batch_is_last in is_last.items():
            if batch_is_last:
                prev_model_state[algo] = None
            else:
                prev_model_state[algo] = model_state[algo].detach()


    def on_train_batch_start(self, batch, batch_idx, unused_optimizers=None):
        torch.compiler.cudagraph_mark_step_begin()

    def training_step(self, batch: DictFeatureBatch, batch_idx: int):
        print(batch[0].keys())
        features, is_first, is_last = batch
        model_state = self.get_model_state(Split.TRAIN, is_first, features)
        (predictions, losses, evaluations), nxt_model_state = self.model(features, model_state)
        self.set_model_state(Split.TRAIN, is_last, nxt_model_state)
        total_loss = self.log_metrics(Split.TRAIN, evaluations, losses, is_first, is_last)
        return total_loss

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        torch.compiler.cudagraph_mark_step_begin()
    
    # This is added to prevent triggering recompilation due to grad mode change
    @torch.enable_grad()
    def validation_step(self, batch: DictFeatureBatch, batch_idx: int):
        features, is_first, is_last = batch
        model_state = self.get_model_state(Split.VAL, is_first, features)
        (predictions, losses, evaluations), nxt_model_state = self.model(features, model_state)
        self.set_model_state(Split.VAL, is_last, nxt_model_state)
        _ = self.log_metrics(Split.VAL, evaluations, losses, is_first, is_last)

    def on_validation_epoch_end(self):
        # reset the model state for validation, this is to ensure that the model state is not carried over from one epoch to the next
        self.val_model_state = {algo: None for algo in self.model.specs.keys()}


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
          val_check_interval: int = 1000,
          wandb_logging: bool = True,
          debug: bool = False,
          compile: bool = False) -> None:

    num_workers = 0 if debug else min(os.cpu_count() - 2, 8)
    project_name = project_name + "_debug" if debug else project_name
    
    pl.seed_everything(config.seed)

    # Dummy call to get the specs
    train_dl = config.get_dataloader(Split.TRAIN, num_workers=num_workers)
    val_dl = config.get_dataloader(Split.VAL, num_workers=0 if debug else 2)        
    model_specs = val_dl.dataset.specs

    model = config.get_model(model_specs)
    if compile:
        print("Compiling model using torch.compile...")
        model.compile()
    else:
        print("Model compilation disabled; skipping torch.compile.")
    
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
                monitor='total/output_score_val',
                save_top_k=3,
                mode='max',
                auto_insert_metric_name=False,
                filename='best-step{step:07d}-Score:{total/output_score_val:.4f}-Loss:{total/loss_val:.4f}',
                wandb_verbose=False
            ),
            ModelCheckpointWithWandbSync(
                wandb_model_suffix="backup",
                monitor='step',
                mode='max',
                save_top_k=2,
                every_n_train_steps=val_check_interval,
                auto_insert_metric_name=False,
                filename='last-step{step:07d}-Score:{total/output_score_val:.4f}-Loss:{total/loss_val:.4f}',
                wandb_verbose=False
            )
        ])

    trainer = pl.Trainer(
        default_root_dir=None,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,   
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16-true" if torch.cuda.is_available() else "32-true",
        devices='auto',
        logger=wandb_logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        max_epochs=-1,
        max_steps=config.num_train_steps,
        limit_train_batches=None,
        limit_val_batches=None,
        check_val_every_n_epoch=None, # Turn off validation  per epoch
        val_check_interval=val_check_interval,
        enable_model_summary=True,
        # detect_anomaly=True
    )

    with trainer.init_module():
        model = TrainingModel(model, config)

    trainer.fit(model, 
                train_dataloaders=train_dl, 
                val_dataloaders=val_dl)




