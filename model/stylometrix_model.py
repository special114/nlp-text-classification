import time
from sklearn.ensemble import RandomForestClassifier
from model.model import Model
import stylo_metrix as sm
import pandas as pd
import numpy as np


DEFAULT_OUTPUT_FILE = 'stylo_model_default.csv'
CHUNK_SIZE = 2000


class StylometrixModel(Model):

    def __init__(self, lang='en', classifier=None, pretrained_path=None, **kwargs):
        self.classifier = classifier if classifier is not None else RandomForestClassifier(max_depth=4, random_state=42)
        self.stylo = sm.StyloMetrix(lang)
        if pretrained_path is not None:
            self.preload_from_file(pretrained_path)

    def preload_from_file(self, file_name):
        x, y = self.load_model(file_name)
        self.classifier.fit(x, y)

    def load_model(self, file_name):
        x_y = pd.read_csv(file_name, delimiter=';')
        x = x_y.loc[:, x_y.columns != 'label'].to_numpy()
        y = x_y['label'].to_numpy()
        return x, y

    def fit(self, texts, labels, greedy=False, save_in_file=False, output_name=None, **kwargs):
        func = self.process_stylo if not greedy else self.process_greedy
        chunks = texts.shape[0] / CHUNK_SIZE
        texts = np.array_split(texts, max(chunks, 1))
        labels = np.array_split(labels, max(chunks, 1))
        for i in range(len(texts)):
            df = func(texts[i], labels[i])
            if save_in_file:
                self.save_df(df, output_name, header=True if i == 0 else False, append=True if i != 0 else False)
            X, y = self.convert_df(df)
            self.classifier.fit(X, y)

    def process_stylo(self, texts, labels=None):
        df = None
        start_time = time.time()
        num_of_errors = 0
        labels = labels.reset_index()['label']

        # liczenie metryk
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print('Row number ', i, ", %s seconds elapsed" % (time.time() - start_time))
            try:
                metrics = self.stylo.transform(text)
                if df is None:
                    df = pd.DataFrame(columns=metrics.columns[1:])
                if labels is not None:
                    metrics.insert(0, 'label', labels[i])
                    df = pd.concat([df, metrics.loc[:, metrics.columns != 'text']], ignore_index=True)
            except:
                num_of_errors += 1
                print('Number of errors: ', num_of_errors)
        print('Number of errors: ', num_of_errors)
        return df

    def process_greedy(self, texts, labels=None):
        metrics = self.stylo.transform(texts)
        data = metrics.loc[:, metrics.columns != 'text']
        data['label'] = labels.reset_index()['label']

        return data

    def save_df(self, df, file_name, header=True, append=False):
        file_name = file_name if file_name is not None else DEFAULT_OUTPUT_FILE
        if df is not None:
            df.to_csv(file_name, sep=';', mode='a' if append else 'w', index=False, header=header)

    def convert_df(self, df):
        X = df.loc[:, df.columns != 'label'].to_numpy()
        if 'label' in df:
            y = df['label'].to_numpy(dtype='int32')
            return X, y
        return X

    def predict(self, X):
        df = self.stylo.transform(X)
        df = df.loc[:, df.columns != 'text']
        return self.classifier.predict(self.convert_df(df))