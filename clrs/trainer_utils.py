import sys
import wandb
import re
import logging
import torch
import os
import re
from typing import List, Optional
from typing_extensions import override
from pathlib import Path
import logging
from pathlib import Path
from rich.table import Table
from rich.console import Console
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

class ModelCheckpointWithWandbSync(ModelCheckpoint):
    """ModelCheckpoint that syncs only current local checkpoints as WandB artifacts.
    
    It tracks only the checkpoints this callback saved (via _save_checkpoint) and when syncing:
      - Deletes any remote artifact whose file no longer exists locally.
      - Uploads new artifacts only for checkpoints recorded by this callback.
      - When an artifact already exists, it updates its best-* alias if needed, but leaves the step alias unchanged.
    """
    def __init__(self, wandb_model_suffix="best", wandb_verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_checkpoints = set()  # Track filenames saved by this callback.
        self._wandb_model_suffix = wandb_model_suffix
        self._artifact_manager = None
        self.wandb_verbose = wandb_verbose
        self.logger = logging.getLogger(__name__)

    def _get_artifact_manager(self, trainer):
        if self._artifact_manager is None:
            # Only initialize artifact manager on rank 0 process
            if trainer.is_global_zero:
                if not hasattr(trainer.logger, 'experiment') or not isinstance(trainer.logger, WandbLogger):
                    self.logger.warning("WandB logger not properly initialized. Skipping artifact sync.")
                    return None
                
                self._artifact_manager = Artifact(
                    entity=trainer.logger.experiment.entity,
                    project_name=trainer.logger.experiment.project,
                    run_name=trainer.logger.experiment.id,
                    verbose=self.wandb_verbose
                )
        return self._artifact_manager

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        filename = os.path.basename(filepath)
        self._saved_checkpoints.add(filename)
        self._sync_wandb_artifacts(trainer)

    def _sync_wandb_artifacts(self, trainer):
        """Sync artifacts only on the main process"""
        if not trainer.is_global_zero:
            return

        artifact_manager = self._get_artifact_manager(trainer)
        if artifact_manager is None:
            return

        checkpoint_dir = Path(self.dirpath)
        # All local checkpoint files (excluding symlinks like "last.ckpt")
        local_ckpts_all = {
            p.name: p for p in checkpoint_dir.glob("*.ckpt")
            if p.is_file() and not p.is_symlink()
        }
        # But only consider for upload those that were saved by this callback.
        local_ckpts_saved = {
            name: path for name, path in local_ckpts_all.items()
            if name in self._saved_checkpoints
        }
        
        try:
            artifact_manager = self._get_artifact_manager(trainer)
            artifacts = artifact_manager.get_artifacts(self._wandb_model_suffix)
            logged_artifacts = {
                artifact.metadata.get("filepath"): artifact 
                for artifact in artifacts 
                if artifact.metadata.get("filepath")
            }

            # Delete remote artifacts for checkpoint files that no longer exist locally.
            for fname, artifact in list(logged_artifacts.items()):
                if fname not in local_ckpts_all:
                    artifact_manager.delete_artifact(artifact)
                    logged_artifacts.pop(fname)

            # Prepare best_k_models using absolute paths.
            best_k_models_abs = {os.path.abspath(str(k)): v for k, v in self.best_k_models.items()}

            # Iterate over this callback's saved checkpoints in ascending order
            for fname, path_obj in sorted(
                local_ckpts_saved.items(),
                key=lambda item: int(re.search(r'step(\d+)', item[0]).group(1))
                        if re.search(r'step(\d+)', item[0]) else 0
            ):
                full_path = os.path.abspath(str(path_obj))
                # Compute best alias if this checkpoint is in best_k_models.
                best_alias = None
                if full_path in best_k_models_abs:
                    sorted_best = sorted(
                        best_k_models_abs.items(),
                        key=lambda x: x[1],
                        reverse=(self.mode == "max")
                    )
                    rank = next(i for i, (p, _) in enumerate(sorted_best) if p == full_path) + 1
                    best_alias = "best" if rank == 1 else f"best-{rank}"

                if fname in logged_artifacts:
                    # Update the artifact's best alias if needed
                    artifact = logged_artifacts[fname]
                    updated_aliases = [a for a in artifact.aliases if not a.startswith("best")]
                    if best_alias:
                        updated_aliases.append(best_alias)
                    artifact_manager.update_artifact_aliases(artifact, updated_aliases)
                else:
                    # Upload new artifact
                    aliases = []
                    if best_alias:
                        aliases.append(best_alias)
                    match = re.search(r'step(\d+)', fname)
                    if match:
                        aliases.append(f"step-{match.group(1)}")
                    
                    artifact_manager.create_and_log_artifact(
                        name=f"model-{trainer.logger.experiment.id}-{self._wandb_model_suffix}",
                        file_path=path_obj,
                        aliases=aliases,
                        metadata={"filepath": fname},
                        run=trainer.logger.experiment
                    )
        except Exception as e:
            self.logger.error(f"Error syncing wandb artifacts: {e}")
            raise


def normalize_state_dict(current_state_dict, ckpt_state_dict):
    # 1. Determine if the checkpoint is compiled by checking its state dict structure
    is_checkpoint_compiled = any("_orig_mod" in key for key in ckpt_state_dict.keys())
    # 2. Determine if our current model is compiled by checking its state dict structure
    is_current_model_compiled = any("_orig_mod" in key for key in current_state_dict.keys())
    
    # 4. Critical fix - ensure checkpoint state dict perfectly matches current model structure
    # but without codebook parameters when there's a mismatch
    new_state_dict = {}
    
    # Get the keys that should be in the final state dict from the current model
    for target_key in current_state_dict.keys():           
        # Find the corresponding key in the checkpoint state dict
        source_key = None
        if is_current_model_compiled and not is_checkpoint_compiled:
            # Current is compiled, checkpoint isn't
            possible_source_key = target_key.replace("model._orig_mod.", "model.")
            if possible_source_key in ckpt_state_dict:
                source_key = possible_source_key
        elif not is_current_model_compiled and is_checkpoint_compiled:
            # Current isn't compiled, checkpoint is
            possible_source_key = target_key.replace("model.", "model._orig_mod.")
            if possible_source_key in ckpt_state_dict:
                source_key = possible_source_key
        else:
            # Both are the same (compiled or not)
            if target_key in ckpt_state_dict:
                source_key = target_key
        
        # If we found a matching source key, copy the value with data type conversion if needed
        if source_key is not None:
            value = ckpt_state_dict[source_key]
            
            # Handle data type conversion if needed
            if hasattr(value, 'dtype') and value.dtype != torch.float32 and value.dtype != torch.int64:
                try:
                    value = value.to(torch.float32)
                except Exception as e:
                    print(f"WARNING: Failed to convert {source_key} from {value.dtype} to float32: {e}")
            
            new_state_dict[target_key] = value

    return new_state_dict



class CustomRichProgressBar(RichProgressBar):
    # -----------------------------
    # Override refresh to catch KeyError in rendering.
    # -----------------------------
    @override
    def refresh(self) -> None:
        try:
            super().refresh()
        except KeyError:
            if self.train_progress_bar_id is not None:
                self._current_task_id = self.train_progress_bar_id
            elif self.val_progress_bar_id is not None:
                self._current_task_id = self.val_progress_bar_id
            else:
                if self.progress and len(self.progress.tasks) > 0:
                    self._current_task_id = 0
            super().refresh()

    # -----------------------------
    # Custom Training Progress Bar
    # -----------------------------
    @override
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._init_progress(trainer)
        if self.progress is not None:
            self.train_progress_bar_id = self.progress.add_task(
                "[green]Training", total=trainer.max_steps
            )
            self._current_task_id = self.train_progress_bar_id
        self.refresh()

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Suppress the default epoch progress bar creation.
        pass

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: any,
        batch_idx: int,
    ) -> None:
        if self.progress is not None and self.train_progress_bar_id is not None:
            self.progress.update(
                self.train_progress_bar_id, completed=trainer.global_step
            )
        self.refresh()

    # -----------------------------
    # Custom Validation Progress Bar
    # -----------------------------
    @override
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._init_progress(trainer)
        # Set the evaluation dataloader index so total_val_batches_current_dataloader works.
        self._current_eval_dataloader_idx = 0
        if not trainer.sanity_checking and self.progress is not None:
            total_val_batches = self.total_val_batches_current_dataloader
            self.val_progress_bar_id = self.progress.add_task(
                f"[blue]Validation Epoch {trainer.current_epoch} (0/{total_val_batches})",
                total=total_val_batches,
                visible=True
            )
            self._current_task_id = self.val_progress_bar_id
        self.refresh()

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled:
            return
        if not trainer.sanity_checking and self.val_progress_bar_id is not None:
            total_batches = self.total_val_batches_current_dataloader
            new_description = f"[blue]Validation Epoch {trainer.current_epoch} ({batch_idx+1}/{total_batches})"
            self.progress.update(
                self.val_progress_bar_id,
                completed=batch_idx + 1,
                description=new_description
            )
        self.refresh()

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.sanity_checking and self.val_progress_bar_id is not None:
            self.progress.update(self.val_progress_bar_id, visible=False)
        self.reset_dataloader_idx_tracker()

    # -----------------------------
    # Custom Sanity-Check Progress Bar
    # -----------------------------
    @override
    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Initialize progress and create a dummy sanity check task.
        self._init_progress(trainer)
        if self.progress is not None:
            # Use total=1 (or adjust if needed) for sanity check.
            self.val_sanity_progress_bar_id = self.progress.add_task(
                "[blue]Sanity Check", total=1, visible=False
            )
            self._current_task_id = self.val_sanity_progress_bar_id
        self.refresh()

    @override
    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Hide the sanity-check progress bar.
        if self.progress is not None and self.val_sanity_progress_bar_id is not None:
            self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
        self.refresh()




class Artifact:
    """Manages W&B model artifacts for checkpoint management."""

    def __init__(self, entity: str, project_name: str, run_name: str, verbose: bool = False):
        self.entity = entity
        self.project_name = project_name
        self.run_name = run_name
        self.run_path = f"{entity}/{project_name}/{run_name}"
        self.console = Console()
        self._api = wandb.Api()
        self._run = None
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self._error_count = 0

    @property
    def run(self):
        """Lazy loading of run object."""
        if self._run is None:
            try:
                self._run = self._api.run(self.run_path)
            except Exception as e:
                raise ValueError(f"Run '{self.run_name}' not found in project '{self.project_name}': {e}")
        return self._run

    def get_artifacts(self, category: str) -> List[wandb.Artifact]:
        """Get all artifacts for a specific category (best/backup)."""
        if category not in ["best", "backup"]:
            raise ValueError("Category must be either 'best' or 'backup'")

        artifacts = []
        for artifact in self.run.logged_artifacts():
            if artifact.type == "model" and f"-{category}" in artifact.name:
                artifacts.append(artifact)
        return artifacts

    def display_checkpoints_table(self, artifacts: List[wandb.Artifact]) -> None:
        """Display available checkpoints in a formatted table."""
        table = Table(
            title=f"\nCheckpoints for run '{self.run_name}'",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Checkpoint", style="dim")
        table.add_column("Step", justify="right", style="green")
        table.add_column("Aliases", style="yellow")

        for artifact in artifacts:
            filename = artifact.metadata.get('filepath', 'unknown')
            step_match = re.match(r'.*step(\d+)-.*\.ckpt', filename)
            
            if step_match:
                step = step_match.group(1)
                table.add_row(
                    filename,
                    str(int(step)),
                    ", ".join(artifact.aliases)
                )
            else:
                table.add_row(
                    filename,
                    "N/A",
                    ", ".join(artifact.aliases)
                )

        self.console.print(table)

    def find_matching_artifact(
        self, 
        artifacts: List[wandb.Artifact], 
        alias: Optional[str]
    ) -> wandb.Artifact:
        """Find artifact matching the given alias.
        
        Args:
            artifacts: List of artifacts to search
            alias: Alias to match (e.g., 'best', 'best-1', 'step-0001000')
            
        Returns:
            Matching artifact
            
        Raises:
            SystemExit: If no matching artifact is found (after displaying available checkpoints)
        """
        if alias is None:
            return None

        for artifact in artifacts:
            if alias in artifact.aliases:
                return artifact

        # If no matching artifact found, show available checkpoints and exit
        print("\nNo matching checkpoint found. Available checkpoints:")
        self.display_checkpoints_table(artifacts)
        sys.exit(1)
        # raise ValueError(f"No checkpoint found with alias '{alias_to_find}'")

    def ensure_local_checkpoint(
        self, 
        artifact: wandb.Artifact, 
        checkpoint_dir: Path
    ) -> Path:
        """Ensure checkpoint exists locally, downloading if necessary."""
        if not artifact.metadata.get("filepath"):
            raise ValueError("Artifact metadata missing filepath information")
        
        checkpoint_filename = artifact.metadata["filepath"]
        local_checkpoint_path = checkpoint_dir / self.project_name / self.run_name / "checkpoints" / checkpoint_filename
        
        if not local_checkpoint_path.exists():
            print(f"Checkpoint not found locally, downloading from W&B...")
            
            local_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            artifact_dir = artifact.download()
            downloaded_checkpoint = next(Path(artifact_dir).glob("*.ckpt"))
            
            downloaded_checkpoint.rename(local_checkpoint_path)
            print(f"Downloaded checkpoint to: {local_checkpoint_path}")
        else:
            print(f"Using existing local checkpoint: {local_checkpoint_path}")

        return local_checkpoint_path 

    def _handle_error(self, error: Exception, context: str) -> None:
        """Handle errors with proper logging and failure counting."""
        self._error_count += 1
        if self._error_count >= 2:
            self.logger.error(f"Failed operation after multiple attempts: {context}")
            raise RuntimeError(f"Failed operation after multiple attempts: {context}") from error
        self.logger.warning(f"Operation failed ({self._error_count}/2): {context}")
        self.logger.warning(f"Error details: {str(error)}")

    def delete_artifact(self, artifact: wandb.Artifact) -> None:
        """Delete an artifact and remove all its aliases."""
        if not getattr(artifact, "id", None):
            self.logger.debug("\nArtifact not published, skipping deletion")
            return
        try:
            artifact.aliases = []
            artifact.save()
            artifact.delete()
            if self.verbose:
                self.logger.info(f"\nDeleted wandb artifact: {artifact.metadata.get('filepath', 'unknown')}")
        except Exception as e:
            self._handle_error(e, f"Error deleting artifact: {artifact.metadata.get('filepath', 'unknown')}")

    def update_artifact_aliases(self, artifact: wandb.Artifact, aliases: List[str]) -> None:
        """Update an artifact's aliases."""
        if set(aliases) != set(artifact.aliases):
            try:
                artifact.aliases = aliases
                artifact.save()
                if self.verbose:
                    self.logger.info(f"\nUpdated artifact aliases to {aliases}")
            except Exception as e:
                self._handle_error(e, f"Error updating artifact aliases to {aliases}")

    def create_and_log_artifact(
        self, 
        name: str, 
        file_path: Path, 
        aliases: List[str],
        metadata: dict,
        run=None
    ) -> wandb.Artifact:
        """Create and log a new artifact with the given file and aliases."""
        try:
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata
            )
            artifact.add_file(str(file_path))
            log_run = run if run is not None else self.run
            log_run.log_artifact(artifact, aliases=aliases)
            if self.verbose:
                self.logger.info(f"\nUploaded wandb artifact for {file_path.name} with aliases {aliases}")
            return artifact
        except Exception as e:
            self._handle_error(e, f"Error creating/logging artifact for {file_path}")
            return None 

    @staticmethod
    def is_alias(value: str) -> bool:
        """Check if a string matches known alias patterns."""
        # Check for known alias patterns
        alias_patterns = [
            r'^best(-\d+)?$',  # best, best-2, best-3, etc.
            r'^latest$',
            r'^step-\d+$'      # step-0000123, etc.
        ]
        return any(re.match(pattern, value) for pattern in alias_patterns)

    @staticmethod
    def is_model_category(value: str) -> bool:
        """Check if a string is a valid model category."""
        return value in ["best", "backup"]

    @staticmethod
    def parse_artifact_string(artifact_string: str, default_project: str) -> tuple[str, str, str, str | None]:
        """Parse an artifact string into its components.
        
        Format: [project/]run_name/{best|backup}[/{alias|step}]
        where:
        - project: Optional project name (defaults to default_project if not specified)
        - run_name: Name of the run
        - best|backup: Model category
        - alias: Optional alias (e.g., 'best', 'best-1', 'latest', 'step-0001000')
        - step: Optional positive integer that will be converted to 'step-NNNNNNN' format
        
        Examples:
            - "run_name/best"  # Lists available checkpoints
            - "run_name/best/best-1"
            - "project/run_name/backup/1000"  # Converted to step-0001000
            - "run_name/best/step-0001000"
        
        Args:
            artifact_string: String in the specified format
            default_project: Default project name to use if not specified
            
        Returns:
            tuple: (project_name, run_name, model_name, alias)
            
        Raises:
            ValueError: If the string format is invalid or ambiguous
        """
        parts = artifact_string.split('/')
        
        if len(parts) < 2:
            raise ValueError(
                "Artifact string must contain at least run_name/model_name. "
                f"Got: {artifact_string}"
            )
            
        if len(parts) == 2:
            # Format: run_name/model_name
            run_name, model_name = parts
            if not Artifact.is_model_category(model_name):
                raise ValueError(f"Invalid model category: {model_name}. Must be 'best' or 'backup'")
            return default_project, run_name, model_name, None
            
        if len(parts) == 3:
            # Could be either:
            # 1. project/run_name/model_name
            # 2. run_name/model_name/alias_or_step
            last_part = parts[2]
            try:
                # Check if the last part is a step number
                step = int(last_part)
                if step <= 0:
                    raise ValueError(f"Step number must be positive, got: {step}")
                # Convert step to alias format
                alias = f"step-{step:07d}"
                run_name, model_name = parts[:2]
                if not Artifact.is_model_category(model_name):
                    raise ValueError(f"Invalid model category: {model_name}. Must be 'best' or 'backup'")
                return default_project, run_name, model_name, alias
            except ValueError:
                # Not a step number, treat as normal alias or project name
                if Artifact.is_alias(last_part):
                    # Case 2: run_name/model_name/alias
                    run_name, model_name, alias = parts
                    if not Artifact.is_model_category(model_name):
                        raise ValueError(f"Invalid model category: {model_name}. Must be 'best' or 'backup'")
                    return default_project, run_name, model_name, alias
                else:
                    # Case 1: project/run_name/model_name
                    project, run_name, model_name = parts
                    if not Artifact.is_model_category(model_name):
                        raise ValueError(f"Invalid model category: {model_name}. Must be 'best' or 'backup'")
                    return project, run_name, model_name, None
                
        if len(parts) == 4:
            # Format: project/run_name/model_name/alias_or_step
            project, run_name, model_name, last_part = parts
            if not Artifact.is_model_category(model_name):
                raise ValueError(f"Invalid model category: {model_name}. Must be 'best' or 'backup'")
            
            try:
                # Check if the last part is a step number
                step = int(last_part)
                if step <= 0:
                    raise ValueError(f"Step number must be positive, got: {step}")
                # Convert step to alias format
                alias = f"step-{step:07d}"
                return project, run_name, model_name, alias
            except ValueError:
                # Not a step number, treat as normal alias
                if not Artifact.is_alias(last_part):
                    raise ValueError(
                        f"Invalid alias format: {last_part}. Must be 'best[-N]', 'latest', 'step-NNNNNN', or a positive integer"
                    )
                return project, run_name, model_name, last_part
            
        raise ValueError(
            f"Invalid artifact string format: {artifact_string}. "
            "Must be [project/]run_name/model_name[/alias_or_step]"
        )

    def get_local_checkpoint(
        self,
        category: str,
        alias: Optional[str],
        checkpoint_dir: Path,
    ) -> Path:
        """Get local checkpoint path for the specified category and alias.
        
        Args:
            category: Category of checkpoint ("best" or "backup")
            alias: Specific alias to load (e.g. "best", "best-1", "step-0001000")
            checkpoint_dir: Directory to store downloaded checkpoints
            
        Returns:
            Path to local checkpoint
            
        Raises:
            ValueError: If artifact cannot be found or loaded
        """
        # Get artifacts for the specified category
        artifacts = self.get_artifacts(category)
        if not artifacts:
            raise ValueError(f"No artifacts found for run '{self.run_name}' in category '{category}'")

        # If no alias specified, list available ones and exit
        if alias is None:
            self.display_checkpoints_table(artifacts)
            sys.exit(0)

        # Find and ensure local checkpoint exists
        matching_artifact = self.find_matching_artifact(artifacts, alias)
        return self.ensure_local_checkpoint(matching_artifact, checkpoint_dir) 