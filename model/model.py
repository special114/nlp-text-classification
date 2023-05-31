from dataclasses import dataclass

@dataclass
class ModelArgs():
    save_in_file: bool
    model_name: str
    pretrainded_path: str


class Model:

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X):
        pass
