"""Tests for trainer initialization without downloading models."""

from torch import nn

from datasets import Dataset
from maskability_index.training.trainer import MaskabilitySeq2SeqTrainer, TrainingPipelineConfig


class TinyConfig:
    """Minimal config object for trainer construction."""

    decoder_start_token_id = 0
    pad_token_id = 0

    def to_dict(self):
        """Return an empty serializable config."""
        return {}


class TinyModel(nn.Module):
    """Tiny torch model compatible with the trainer constructor."""

    config = TinyConfig()
    main_input_name = "input_ids"

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Return a deterministic zero loss."""
        return {"loss": input_ids.float().sum() * 0}

    def prepare_decoder_input_ids_from_labels(self, labels):
        """Mirror labels for collator compatibility."""
        return labels


class TinyTokenizer:
    """Tiny tokenizer compatible with the data collator."""

    pad_token_id = 0
    model_input_names = ["input_ids", "attention_mask"]

    def batch_decode(self, values, skip_special_tokens=True):
        """Decode all rows to empty strings."""
        return ["" for _ in values]

    def save_pretrained(self, save_directory):
        """Pretend to save tokenizer assets."""
        return (save_directory,)


def test_trainer_initialization(tmp_path) -> None:
    """Trainer initialization should wire output directory into training args."""
    dataset = Dataset.from_list(
        [{"input_ids": [1], "attention_mask": [1], "labels": [1]}]
    )
    trainer = MaskabilitySeq2SeqTrainer(
        model=TinyModel(),
        tokenizer=TinyTokenizer(),
        train_dataset=dataset,
        eval_dataset=dataset,
        config=TrainingPipelineConfig(output_dir=str(tmp_path), epochs=1, batch_size=1),
    )
    assert trainer.trainer.args.output_dir == str(tmp_path)
