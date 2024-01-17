import datasets
import evaluate
import torch
import numpy as np


# Modified from https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
def main_preprocess_function(examples, tokenizer, text_field, prompt_begin, prompt_end, label_field, sequence_len, single_token=True):
    batch_size = len(examples[text_field])

    # Prepare the context with the text in between of prompts, e.g. "Sentence : <text> Label :"
    inputs = [prompt_begin + x + prompt_end for x in examples[text_field]]
    
    # Prepare the prediction part
    targets = [str(x) for x in examples[label_field]]
    
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    
    # Concatenate the context and prediction parts as one input and set -100 to the labels of the context part
    # This is because only the label part will be used to calculate the loss
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        if single_token:
            # Tokenizer adds <s> to input_ids so just take the last id
            # NOTE THAT THIS ASSUMES THE LABEL IS SINGLE TOKEN
            label_input_ids = [labels["input_ids"][i][-1]]
        else:
            # Tokenizer adds <s> to input_ids so just take the rest
            label_input_ids = labels["input_ids"][i][1:]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # Pad the samples with sequence_len and trim if longer than sequence_len
    # NOTE THAT IF CONTEXT IS LONGER THAN SEQUENCE_LEN, THERE WILL BE NOTHING TO PREDICT, LABEL IS ALL -100
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            sequence_len - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (sequence_len - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (sequence_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:sequence_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:sequence_len])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:sequence_len])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class Dataset:
    dataset = None
    classes = None # List of class labels
    text_field = None # Name of the field in the dataset that contains the text
    prompt_begin = None # Prompt to add to the beginning of the text, e.g. "Sentence : "
    prompt_end = None # Prompt to add to the end of the text, e.g. " Label :"
    label_field = None # Name of the field in the dataset that contains the label
    evaluate = None # Evaluation metric
    run_test = False # Whether to run test set evaluation

    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def target_max_len(self):
        target_lens = [len(self.tokenizer(class_label)["input_ids"]) for class_label in self.classes]
        target_max_len = max(target_lens)
        return target_max_len

    def preprocess_function(self, example):
        return main_preprocess_function(example, self.tokenizer, self.text_field, self.prompt_begin,
                                         self.prompt_end, self.label_field, self.sequence_len)
    
    # Define the evaluation metric (NOTE THAT THIS ASSUMES THE LABEL IS SINGLE TOKEN)
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Only keep predictions for the last token shifted by 1
        predictions = predictions[..., -2]
        # Only keep labels for the last token
        labels = labels[..., -1]
        return self.evaluate.compute(predictions=predictions, references=labels)

    def preprocess_logits_for_metrics(self, logits, labels):
        """
        Original Trainer may lead to a memory issue.
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids


class SST2Dataset(Dataset):
    def __init__(self, tokenizer, sequence_len):
        # Load data
        self.dataset = datasets.load_dataset('sst2')
        # Map labels from 0/1 to negative/positive
        self.classes = ['negative', 'positive']
        self.dataset = self.dataset.map(
            lambda x: {"text_label": [self.classes[label] for label in x["label"]]},
            batched=True,
            num_proc=8,
        )
        self.text_field = "sentence"
        self.prompt_begin = "Sentence : "
        self.prompt_end = " Label :"
        self.label_field = "text_label"
        self.evaluate = evaluate.load("accuracy")
        super().__init__(tokenizer, sequence_len)


class QNLIDataset(Dataset):
    def __init__(self, tokenizer, sequence_len):
        # Load data
        self.dataset = datasets.load_dataset('glue', 'qnli')
        self.classes = ['0', '1']
        self.dataset = self.dataset.map(
            lambda x: {"text_concat": [question + " ### " + sentence for question, sentence in zip(x["question"], x["sentence"])]},
            batched=True,
            num_proc=8,
        )
        # 5k eval samples too large, shuffle and reduce it to 1k
        self.dataset['validation'] = self.dataset['validation'].shuffle().select(range(1000))
        self.text_field = "text_concat"
        self.prompt_begin = "Two sentences separated with ### : "
        self.prompt_end = " Label :"
        self.label_field = "label"
        self.evaluate = evaluate.load("accuracy")
        super().__init__(tokenizer, sequence_len)


class CNNDataset(Dataset):
    def __init__(self, tokenizer, sequence_len):
        # Load data
        self.dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")
        # 13.4k eval samples too large, shuffle and reduce it to 1k
        self.dataset['validation'] = self.dataset['validation'].shuffle().select(range(1000))
        # Get rid of the test dataset
        del self.dataset['test']
        self.text_field = "article"
        self.prompt_begin = "Article : "
        self.prompt_end = " Summary :"
        self.label_field = "highlights"
        self.evaluate = evaluate.load("rouge")
        self.run_test = True
        super().__init__(tokenizer, sequence_len)

    def preprocess_function(self, example):
        return main_preprocess_function(example, self.tokenizer, self.text_field, self.prompt_begin,
                                         self.prompt_end, self.label_field, self.sequence_len, single_token=False)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        # Only keep predictions up to last token
        predictions = predictions[..., :-1]
        # Only keep labels from the first token
        labels = labels[..., 1:]
        # Replace -100 of the labels as we don't want the content
        predictions = np.where(labels != -100, predictions, self.tokenizer.pad_token_id)
        # Decode generated summaries into text
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute ROUGE scores
        result = self.evaluate.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        return {k: round(v, 4) for k, v in result.items()}

    def compute_test_metrics(self, trainer):
        test_dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split='test')
        # Filter out samples too long, e.g. more than 500 words
        test_dataset = test_dataset.filter(lambda x: len(x['article'].split()) < 500)
        # 11.4k test samples too large, shuffle and reduce it to 1k
        test_dataset = test_dataset.shuffle().select(range(1000))
        # Add prompt_begin and prompt_end
        test_dataset = test_dataset.map(
            lambda x: {"article": [self.prompt_begin + article + self.prompt_end for article in x["article"]]},
            batched=True,
            num_proc=None,
        )

        # Tokenize data
        def test_preprocess_function(examples):
            model_inputs = trainer.tokenizer(examples['article'], padding=False)

            # 2. reserve the original article and summary for saving
            model_inputs['summary'] = examples['highlights']
            return model_inputs

        with trainer.args.main_process_first(desc="tokenizing test dataset"):
            test_dataset = test_dataset.map(
                test_preprocess_function,
                batched=True, num_proc=None, desc="tokenizing dataset",
                remove_columns=test_dataset.column_names)

        # Filter out samples too long, e.g. more than 750 tokens
        test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) < 750)

        test_dataset.set_format(type="torch")

        def generate_batched(
            model,
            tokenizer,
            device,
            query_tensors,
            batch_size: int = 4,
            return_prompt: bool = True,
            pad_to_multiple_of: int = None,
            **generation_kwargs,
        ):
            outputs = []

            tokenizer.padding_side = "left"

            # handle distributed case and distribute query_tensors among gpus
            query_tensors = query_tensors[device.index::trainer.args.world_size]

            # in case we have fewer examples than bs
            batch_size = min(len(query_tensors), batch_size)

            for i in range(0, len(query_tensors), batch_size):
                # prevent overflow if query tensors are not even multiple of bs
                end_index = min(len(query_tensors), i + batch_size)

                batch = query_tensors[i:end_index]
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {"input_ids": batch, "attention_mask": batch_mask}

                padded_inputs = tokenizer.pad(
                    inputs,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    generations = model.generate(**padded_inputs, **generation_kwargs)

                for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                    output = generation[(1 - mask).sum() :]  # remove padding

                    if not return_prompt:
                        output = output[(mask).sum() :]  # remove prompt
                    outputs.append(output)

            return outputs

        if hasattr(trainer.model, "generate"):
            model = trainer.model
        # The following is for GradSampleModule wrapping
        elif hasattr(trainer.model._module, "generate"):
            model = trainer.model._module
        # The following is for GradSampleModule and DPDDP wrapping
        elif hasattr(trainer.model._module.module, "generate"):
            model = trainer.model._module.module
        else:
            raise ValueError("Cannot find generate function in the model.")
    
        model.eval()
        generation_kwargs = {"max_new_tokens": 100, "pad_token_id": trainer.tokenizer.pad_token_id, 
                             "eos_token_id": trainer.tokenizer.eos_token_id,}

        response_tensors = generate_batched(
            model, trainer.tokenizer, trainer.args.device,
            test_dataset["input_ids"],
            batch_size=trainer.args.eval_batch_size, return_prompt=False,
            **generation_kwargs
        )

        responses = [trainer.tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                                    for r in response_tensors]

        result = self.evaluate.compute(
            predictions=responses, references=test_dataset["summary"][trainer.args.device.index::trainer.args.world_size],
            use_stemmer=True
        )

        r1 = trainer.accelerator.reduce(torch.tensor(result['rouge1']).to(trainer.args.device), reduction="mean")
        r2 = trainer.accelerator.reduce(torch.tensor(result['rouge2']).to(trainer.args.device), reduction="mean")
        rl = trainer.accelerator.reduce(torch.tensor(result['rougeL']).to(trainer.args.device), reduction="mean")

        result = {'rouge1': r1.item(), 'rouge2': r2.item(), 'rougeL': rl.item()}
        return {k: round(v, 4) for k, v in result.items()}


ALL_DATASETS = {"sst2": SST2Dataset, "qnli": QNLIDataset, "cnn": CNNDataset}