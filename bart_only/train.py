import nltk
import numpy as np
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric


if __name__ == "__main__":
    dataset = datasets.load_dataset("ccdv/govreport-summarization")
    # dataset = datasets.load_dataset("ccdv/pubmed-summarization")
    # dataset = datasets.load_dataset("ccdv/arxiv-summarization")
    data_train = dataset['train']
    data_validation = dataset['validation']
    data_test = dataset['test']

    print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Training Started!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

    max_input_length = 1024
    max_target_length = 1024

    def preprocess_function(examples):
        inputs = examples["report"]
        summary = examples["summary"]
        if len(inputs) == 0 or len(summary) == 0:
            pass
        else:
            model_inputs = tokenizer(inputs, padding="max_length", max_length=max_input_length, truncation=True, return_tensors="pt")

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(summary, padding="max_length", max_length=max_target_length, truncation=True, return_tensors="pt")

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

    batch_size = 2
    args = Seq2SeqTrainingArguments(
        "bart-large-finetuned-govreport",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Training Finished!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
