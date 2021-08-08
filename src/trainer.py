from typing import Optional, Dict, List
import wandb
from torch.utils.data import Dataset

from transformers import trainer as trainer_script
from transformers.utils import logging
from transformers.integrations import (
    WandbCallback,
    is_wandb_available,
    TensorBoardCallback,
    CometCallback,
    AzureMLCallback,
    MLflowCallback,
)

from funnel_vae.src.trainer_callback import WandbCallbackUseModelLogs, TellModelGlobalStep

logger = logging.get_logger(__name__)

assert(is_wandb_available())

NOT_ALLOWED_LOGGERS = [TensorBoardCallback, CometCallback, AzureMLCallback, MLflowCallback]

for logger_integration in NOT_ALLOWED_LOGGERS:
    removed = []
    if logger_integration in trainer_script.DEFAULT_CALLBACKS:
        trainer_script.DEFAULT_CALLBACKS.remove(logger_integration)
        removed.append(logger_integration)
    logger.info(f"Only supports W&B logging, removed loggers: {removed}")


class VaeTrainer(trainer_script.Trainer):
    text_to_array = None

    def __init__(self, model=None, args=None, custom_methods={}, **kwargs):
        self.clean_tkn_spaces = not args.dont_clean_up_tokenization_spaces
        if args.render_text_image:
            assert 'custom_text_to_array' in custom_methods
            self.text_to_array = custom_methods['custom_text_to_array']
        super().__init__(model, args, **kwargs)
        self.remove_callback(WandbCallback)
        self.add_callback(WandbCallbackUseModelLogs)
        self.add_callback(TellModelGlobalStep)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        Adds extra VAE tests:
        - Interpolation between samples in latent space.
        """
        if self.state.global_step < wandb.run.history._step:
            self.state.global_step = wandb.run.history._step
        output_metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        return output_metrics
