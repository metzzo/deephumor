import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utility.load_class import class_for_name
from utility.pytorch_dataset import PyTorchDataset
import pandas as pd
import numpy as np


class BaseDataset(object):
    @property
    def train_dataset(self):
        raise NotImplementedError()

    @property
    def class_column_name(self):
        raise NotImplementedError()

    @property
    def trafos(self):
        return {}

    @property
    def num_classes(self):
        return self.train_label.max()[0] + 1

    @property
    def num_input(self):
        return len(self.train_data.columns)

    def load(self):
        data = pd.read_csv(
            sep=',',
            quotechar='"',
            filepath_or_buffer=os.path.join(self.args.directory, self.train_dataset + '.csv'),
        )
        if 'ID' in data.columns:
            data.drop(labels=['ID'], inplace=True, axis=1)

        if 'has_null' in data.columns:
            data = data[data['has_null'] == 0]

        data = data.replace(np.nan, 'Unknown', regex=True)
        self.shuffle(data)

        label = pd.DataFrame({self.class_column_name: data[self.class_column_name]})
        data.drop(
            labels=[self.class_column_name],
            inplace=True,
            axis=1
        )

        data = self.apply_transforms(trafos=self.trafos, data=data)

        split = self.get_split(data)
        self.train_data = data[:split]
        self.validation_data = data[split:]
        self.train_label = label[:split]
        self.validation_label = label[split:]

        self.train_label, self.validation_label = self.convert_categorical_labels(
            categorical_labels=[self.class_column_name],
            train_data=self.train_label,
            validation_data=self.validation_label
        )

    def __init__(self, args):
        self.args = args
        self.classes = None
        self.train_data, self.validation_data, self.train_label, self.validation_label = [None] * 4
        self.test_data = None
        self.test_ids = None

        self.encoders = {}
        self.scalers = {}

        self.scaler_class = class_for_name('sklearn.preprocessing', args.scaler)
        self.load()

    def pytorch_dataset(self, usage):
        data, labels = self.get_subset(usage=usage)
        return PyTorchDataset(
            data=data,
            labels=labels,
        )

    def get_split(self, data):
        multiplicator = self.args.train_ratio if self.args.validate else 1.0
        return int(len(data) * multiplicator)

    def get_subset(self, usage):
        if usage == 'train':
            return (
                self.train_data,
                self.train_label,
            )
        elif usage == 'val':
            return (
                self.validation_data,
                self.validation_label,
            )
        elif usage == 'test':
            return (
                self.test_data,
                self.test_ids
            )
        else:
            raise RuntimeError("Invalid usage")

    def shuffle(self, df):
        df.sample(frac=1).reset_index(drop=True)

    def categorical_to_numerical(self, series):
        encoder = LabelEncoder()
        encoder.fit(series)
        return encoder, encoder.transform(series)

    def categorical_to_one_hot(self, series):
        values = series.values.reshape(-1, 1)

        encoder = OneHotEncoder(
            sparse=False,
            handle_unknown='ignore'
        )
        encoder.fit(values)
        return encoder, encoder.transform(values)

    def convert_categorical_labels(self, categorical_labels, train_data, validation_data):
        categorical_labels = list(filter(lambda x: len(x) > 0, categorical_labels))

        results = []
        val_results = []
        for label in categorical_labels:
            encoder, numerical_data = self.categorical_to_numerical(train_data[label])
            self.encoders[label] = encoder
            results.append(
                pd.DataFrame({label: numerical_data})
            )
            val_results.append(
                pd.DataFrame({
                    label: encoder.transform(
                        validation_data[label]
                    )
                })
            )
        return pd.concat(results, axis=1), pd.concat(val_results, axis=1)

    def apply_transforms(self, trafos, data):
        if len(trafos) > 0:
            results = []
            for column in data.columns:
                result = data[column]
                if column in trafos:
                    result = result.apply(trafos[column])
                results.append(result)
            for column, trafo in trafos.items():
                if column not in data.columns:
                    results.append(pd.DataFrame(data.apply(trafo, axis=1), columns=[column,]))
            return pd.concat(results, axis=1)
        else:
            return data

    def convert_one_hot_labels(self, categorical_labels, train_data, validation_data):
        categorical_labels = list(filter(lambda x: len(x) > 0, categorical_labels))

        results = []
        val_results = []
        for label in categorical_labels:
            encoder, numerical_data = self.categorical_to_one_hot(train_data[label])
            self.encoders[label] = encoder
            columns = list(
                map(
                    lambda x: '{0}_{1}'.format(label, x),
                    range(0, len(numerical_data[0]))
                )
            )

            results.append(
                pd.DataFrame(numerical_data, columns=columns)
            )
            val_results.append(
                pd.DataFrame(
                    encoder.transform(
                        validation_data[label].values.reshape(-1, 1)
                    ),
                    columns=columns
                )
            )
        return pd.concat(results, axis=1), pd.concat(val_results, axis=1)

    def scale_columns(self, scaling_columns, train_data, validation_data, test_data=None):
        scaling_columns = list(filter(lambda x: len(x) > 0, scaling_columns))
        scaler = self.scaler_class()

        train_to_normalize = train_data[scaling_columns]
        validation_to_normalize = validation_data[scaling_columns]
        if test_data is not None:
            test_to_normalize = test_data[scaling_columns]
        else:
            test_to_normalize = None

        for column in scaling_columns:
            self.scalers[column] = scaler
            #train_to_normalize[column] = train_to_normalize[column].astype(float)
            #validation_to_normalize[column] = validation_to_normalize[column].astype(float)

        scaler.fit(train_to_normalize)

        return [pd.DataFrame(
            scaler.transform(train_to_normalize),
            columns=scaling_columns
        ), pd.DataFrame(
            scaler.transform(validation_to_normalize),
            columns=scaling_columns
        )] + ([pd.DataFrame(
                scaler.transform(test_to_normalize),
                columns=scaling_columns
            )
        ] if test_data is not None else [])
