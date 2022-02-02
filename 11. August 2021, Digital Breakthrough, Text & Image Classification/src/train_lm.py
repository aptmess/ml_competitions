"""
Prerain Distilbert as masked language model on all item names.
"""

from pathlib import Path
from transformers import (DataCollatorForLanguageModeling,
                          DistilBertConfig,
                          DistilBertForMaskedLM,
                          LineByLineTextDataset,
                          PreTrainedTokenizerFast,
                          Trainer,
                          TrainingArguments,
                          BertConfig,
                          BertForMaskedLM,
                          XLMRobertaConfig,
                          XLMRobertaForMaskedLM,
                          RobertaConfig,
                          RobertaForMaskedLM)


class LMTokenizer(object):

    def __init__(self,
                 tokenizer_path,
                 tokenizer=None,
                 mask_token=None,
                 pad_token=None,
                 sep_token=None,
                 cls_token=None,
                 unk_token=None):
        if tokenizer is None:
            self._tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path
            )
        else:
            self._tokenizer = tokenizer
        self._tokenizer.mask_token = '[MASK]' or mask_token
        self._tokenizer.pad_token = "[PAD]" or pad_token
        self._tokenizer.sep_token = "[SEP]" or sep_token
        self._tokenizer.cls_token = "[CLS]" or cls_token
        self._tokenizer.unk_token = "[UNK]" or unk_token

    @property
    def load(self):
        return self._tokenizer


class LMModel(object):

    def __init__(self, model_name, **kwargs):
        if model_name == 'DISTILBERT':
            self._config = DistilBertConfig(**kwargs)
            self._model = DistilBertForMaskedLM(self._config)
        elif model_name == 'BERT':
            self._config = BertConfig(**kwargs)
            self._model = BertForMaskedLM(self._config)
        elif model_name == 'XLMROBERTA':
            self._config = XLMRobertaConfig(**kwargs)
            self._model = XLMRobertaForMaskedLM(self._config)
        elif model_name == 'ROBERTA':
            self._config = RobertaConfig(**kwargs)
            self._model = RobertaForMaskedLM(self._config)
        else:
            raise ValueError(f'unknown model name {model_name}')

    @property
    def load(self):
        return self._model


class LMDataset(object):
    def __init__(self,
                 tokenizer,
                 file_path,
                 **kwargs):
        self._dataset = LineByLineTextDataset(
            tokenizer=tokenizer.load,
            file_path=file_path,
            **kwargs
        )

    @property
    def load(self):
        return self._dataset


class LMDataCollator(object):
    def __init__(self, tokenizer, **kwargs):
        self._collator = DataCollatorForLanguageModeling(
            tokenizer.load,
            **kwargs
        )

    @property
    def load(self):
        return self._collator


class LMTrainingArgs(object):
    def __init__(self, output_dir, **kwargs):
        self._training_args = TrainingArguments(
            output_dir,
            **kwargs
        )

    @property
    def load(self):
        return self._training_args


class LMTrainer(object):

    def __init__(self,
                 model: LMModel,
                 training_args: LMTrainingArgs,
                 data_collator: LMDataCollator,
                 train_dataset: LMDataset,
                 eval_dataset: LMDataset,
                 **kwargs):
        self._trainer = Trainer(
            model=model.load,
            args=training_args.load,
            data_collator=data_collator.load,
            train_dataset=train_dataset.load,
            eval_dataset=eval_dataset.load,
            **kwargs
        )
        print(eval_dataset.load)
        self.fitted = False

    def fit(self):
        self._trainer.train()
        self.fitted = True

    def save_model(self,
                   output_path: Path,
                   name: str):
        if self.fitted:
            self._trainer.save_model(str(output_path / name))
        else:
            raise ValueError('Fit tokenizer before saving')
