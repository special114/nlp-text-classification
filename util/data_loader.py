import pandas as pd

from util.label_encoder import LabelEncoder


class DataLoader():
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def get_train_data(self, train_data_file=None, train_data_count=None, **kwargs):
        return self.get_data(specific_data_file=train_data_file, data_label='train', specific_data_count=train_data_count, **kwargs)

    def get_test_data(self, test_data_file=None, test_data_count=None, **kwargs):
        return self.get_data(specific_data_file=test_data_file, data_label='test', specific_data_count=test_data_count, **kwargs)

    def get_data(self, specific_data_file=None, data_file=None, data_label=None, specific_data_count=None, data_count=None, **kwargs):
        file = specific_data_file if specific_data_file is not None else data_file
        if file is None:
            raise Exception(f'No {data_label} data file provided')
        
        if specific_data_file is not None:
            data_label=None
        
        count = specific_data_count if specific_data_count is not None else data_count
        if count is not None:
            count = int(count)

        return self.load_data(file_name=file, data_label=data_label, count=count, **kwargs)

    def load_data(self, file_name=None, data_label=None, text_column_name='text', label_column_name='label', count=None, **kwargs):
        data = pd.read_csv(file_name)
        if data_label is not None:
            data = data.loc[data['split'] == data_label].reset_index()
        df = data[[text_column_name, label_column_name]].rename(columns={text_column_name: 'text', label_column_name: 'label'})
        if count is not None:
            df = df.head(count)
        y = self.label_encoder.encode(df['label'])
        return df['text'], y