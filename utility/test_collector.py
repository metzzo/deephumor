import pandas as pd


class TestCollector(object):
    def __init__(self, dataset):
        self.dataset = dataset

        expected_size = len(self.dataset.test_data)
        self.df = pd.DataFrame(columns=('ID', 'Class'), index=range(expected_size))
        self.index = 0

    def collect_test(self, id, predicted_class):
        predicted_class = self.dataset.encoders[
                self.dataset.class_column_name
            ].inverse_transform(predicted_class)

        for id, predicted_class in zip(id, predicted_class):
            self.df.iloc[self.index]['ID'] = id
            self.df.iloc[self.index]['Class'] = predicted_class

            self.index += 1

    def save_csv(self, path):
        self.df['ID'] = self.df['ID'].astype(int)
        self.df.to_csv(
            path_or_buf=path,
            columns=('ID', 'Class',),
            index=False,
        )
