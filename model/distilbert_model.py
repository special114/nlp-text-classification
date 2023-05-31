import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, pipeline

from model.model import Model


DEFAULT_MODEL_NAME = 'bert_model'


class DistilBertModel(Model):
    def __init__(self, pretrained_path='distilbert-base-uncased', **kwargs):
        self.device = None
        self.pretrained_path = pretrained_path if pretrained_path is not None else 'distilbert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        self.id2label = None
        self.label2id = None
        self.model = None

    def fit(self, X, y, output_name=None, **kwargs):
        tokenized_data = X.apply(lambda x: self.tokenizer(x, truncation=True))
        tokenized_data_df = pd.concat([tokenized_data.apply(lambda x: x.input_ids).rename('input_ids'),
                                       tokenized_data.apply(lambda x: x.attention_mask).rename('attention_mask')],
                                      axis=1)

        train = pd.concat([X, tokenized_data_df], axis=1)
        train['label'] = y.tolist()
        train = train.to_dict('records')

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.id2label = {int(x): 'LABEL_' + str(x) for x in np.unique(y)}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_path, id2label=self.id2label, label2id=self.label2id
        ) 

        training_args = TrainingArguments(
            output_dir=output_name if output_name is not None else DEFAULT_MODEL_NAME,
            # learning_rate=2e-5,
            # per_device_train_batch_size=16,
            # per_device_eval_batch_size=16,
            # num_train_epochs=2,
            # weight_decay=0.01,
            # # evaluation_strategy="epoch",
            save_strategy="epoch",
            # load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            # eval_dataset=test,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
        )

        self.device = trainer.model.device

        trainer.train()


    def predict(self, X):
        classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
        score = X.apply(classifier)
        y_pred = score.apply(lambda x: x[0]['label'])\
            .apply(lambda x: self.label2id[x])
        return y_pred
