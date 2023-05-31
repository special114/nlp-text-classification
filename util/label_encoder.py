class LabelEncoder():
    def __init__(self):
        self.label_map = {}
        self.counter = 0

    def encode(self, df):
        return df.apply(self.get_mapping)
    
    def get_mapping(self, label):
        if label not in self.label_map:
            self.label_map[label] = self.counter
            self.counter += 1
        return self.label_map[label]