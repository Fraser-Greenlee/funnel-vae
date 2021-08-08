
import logging
from dataclasses import dataclass, field

from transformers import (
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class VaeTrainingArguments(TrainingArguments):
    """
    Extra arguments to specify generation during evaluation.
    """
    save_steps: int = field(default=1000, metadata={"help": "Save checkpoint every X updates steps."})
    generate_min_len: int = field(
        default=1,
        metadata={"help": "The minimum length of sequences to be generated from latent points during evaluation."},
    )
    generate_max_len: int = field(
        default=20,
        metadata={"help": "The maximum length of sequences to be generated from latent points during evaluation."},
    )
    seq_check: str = field(
        default=None,
        metadata={"help": f"Run check on sequences from random latent codes. Options: {', '.join([str(k) for k in SEQ_CHECKS.keys()])}"},
    )
    max_validation_size: int = field(
        default=None,
        metadata={"help": "Limit the eval dataset size, defaults to not limiting it, must be < validation size."},
    )
    sample_from_latent: bool = field(
        default=False,
        metadata={"help": "Whether to sample from the latent space during evaluation."},
    )
    test_classification: bool = field(
        default=False,
        metadata={"help": "Test using latent codes for unsupervised classification."},
    )
    advisery_weight: int = field(
        default=1,
        metadata={"help": "Encourage the encoder & decoder to produce a bijective mapping. Feeds the final decoder hidden state to the encoder and compares the latent codes."},
    )
    interpolate_training_step_rate: int = field(
        default=1,
        metadata={"help": "Run a batch of iterpolation losses every N steps."},
    )
    render_text_image: bool = field(
        default=False,
        metadata={"help": """Render sequence as an image and log it to Weights & Biasis during interpolations.
                            Must be using a dataset with array_to_text & text_to_array methods.
                            This will override seq_check & just see if the text is a valid image."""},
    )
    dont_clean_up_tokenization_spaces: bool = field(
        default=False,
        metadata={"help": "Don't clean up token spaces, turn off for non NLP tasks."},
    )
    interpolate_all_at_once: bool = field(
        default=False,
        metadata={"help": "Treat all latent tokens as one during slerp."},
    )
    n_tokenized_segments: int = field(default=0, metadata={"help": "Don't train just tokenize dataset into N segments instead."})
