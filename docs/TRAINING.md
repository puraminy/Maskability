# Training Pipeline

Milestone 3 implements a reusable HuggingFace seq2seq training pipeline. The training code only fine-tunes language models; it does not compute DepthRank or the Maskability Index.

## Pipeline

1. Load `RelationInstance` triples from the dataset module.
2. Build prompts with the configured prompt builder (`prefix`, `masked`, or future builders).
3. Convert examples to HuggingFace `Dataset` or `DatasetDict` objects.
4. Tokenize `input_text` and `target_text` with configurable maximum lengths.
5. Train with `Seq2SeqTrainer` and deterministic seeds.
6. Evaluate and save the resulting checkpoint.

## Checkpoints

`MaskabilitySeq2SeqTrainer.save_checkpoint()` writes the model, tokenizer assets, and trainer state to the configured output directory. Training can resume by passing a checkpoint path, `True`, or the latest checkpoint semantics supported by HuggingFace to `train(resume_from_checkpoint=...)`. Existing checkpoints can be loaded into a trainer wrapper with `load_checkpoint()`.

## Configuration

Hydra configuration for Milestone 3 lives in `configs/experiment/milestone3.yaml`. It exposes:

- model name and tokenizer name
- epochs
- batch size
- learning rate
- optimizer
- scheduler
- warmup steps
- weight decay
- max input length
- max target length
- generation length
- beam size
- seed
- mixed precision
- output directory
- tokenized dataset cache directory

Run the experiment with:

```bash
python experiments/train_model.py
```

Override any field with Hydra syntax, for example:

```bash
python experiments/train_model.py experiment.model.name=google/t5-base experiment.training.batch_size=4
```

## Extending to other models

Model loading is isolated in `src/maskability_index/models/factory.py`. Add a new `Seq2SeqModelSpec` to a `Seq2SeqModelFactory` instance or to the default registry. Training code depends only on the factory interface and does not need to change when a new seq2seq model is registered.
