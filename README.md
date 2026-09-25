# LabAI-HPCTools
## Deliverable 1
### Key contents
### Hugging face repo (link to it PUT IT)
#### run_qa.py
File in charge of the full workflow of the training of BERT-uncased model (base model) with SQUAD dataset. The script gives support to different training configurations within several arguments programmed with `HfArgumentParser` class, for this purpose 3 python classes are created: 
- `ModelArguments`: to select the model and tokenizer and their configuration
- `DataTrainingArguments`: configuration of the training dataset, in this case, SQUAD v1.0
- `TrainingArguments`: is imported from transformers library needed as a requirement from source repo. 

```python
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
```
##### 1. Loading dataset and model
This script programs automatically training with SQUAD dataset in their own `datasets` library (downloaded as a requirement) so it is loaded in this point of the code. It also gives support for using personal datasets which is programmed in the other if branch.
```python
if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
```
Regarding the model, config extracts the arguments charged before and the model and tokenizer are instantiated:
```python
config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
```
##### 2. Preprocessing
In this stage, the script transforms the questions and paragraphs in BERT-format. The function `def prepare_train_features(examples)` is in charge of that.

##### 3. Training and evaluation
We use `QuestionAnsweringTrainer` an adaptation of `Trainer` class of `transformers`library, made by HuggingFace in this repo. (`trainer_qa.py` file).
`trainer.train()` executes the training with the hardware resources provided by the job `run_baseline.sh`.
```python
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint) <- TRAINING STARTS
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
```
Finally, the model evaluates its performance in the validation dataset with trainer.evaluate()
```python
# Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
```
The rest of the file is in charge of a kind of a post-processing of the predictions.

#### trainer_qa.py

#### utils_qa.py

### SLURM jobs
#### run_baseline.sh

#### run_profiling.sh


### Reporting training times


### Profiling
