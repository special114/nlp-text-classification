from sklearn.metrics import accuracy_score, f1_score

from model.distilbert_model import DistilBertModel
from model.stylometrix_model import StylometrixModel
from util.argsparse import get_model_args
from util.data_loader import DataLoader

import numpy as np


MODELS = {
    'stylo': StylometrixModel,
    'bert': DistilBertModel
}


class Processor():
    def __init__(self, model_args):
        self.model_args = model_args
        self.model = self.create_model(model_args)
        self.data_loader = DataLoader()

    def create_model(self, model_args):
        create_func = MODELS.get(model_args['model'])
        if create_func is None:
            raise 'Invalid model name: ' + model_args['model']
        model = create_func(**model_args)
        return model
    
    def process(self):
        if self.model_args.get('pretrained_path') is None:
            self.train()
        self.score()
    
    def train(self):
        text, labels = self.data_loader.get_train_data(**self.model_args)
        self.model.fit(text, labels, **self.model_args)
    
    def score(self):
        x, y = self.data_loader.get_test_data(**self.model_args)
        y_pred = self.model.predict(x)
        np.savetxt("output.csv", y_pred, fmt='%d')
        print(accuracy_score(y, y_pred))
        print(f1_score(y, y_pred, average='macro'))
        

def main():
    args = get_model_args()
    Processor(args).process()


if __name__ == '__main__':
    main()
