from dataset.base import BaseDataset
import pandas as pd


import datetime

class JigsawDataset(BaseDataset):
    @property
    def train_dataset(self):
        return 'jigsaw/jigsaw'

    @property
    def class_column_name(self):
        return 'target'


    def load(self):
        super(JigsawDataset, self).load()

        one_hot_labels = (
            'Region',
            'Crime',
            'Victim Race',
            'County',
            'State',
            'Method',
        )
        categorical_labels = [
            'Sex',
            'Juvenile',
            'Volunteer',
            'Federal',
            'Foreign National',
        ]
        scaling_labels = [
            'Male Victim',
            'Female Victim',
            'weekday',
            'month',
            'year',
            'day_of_year',
            'week_number_of_year',
            'Age',
            'Victim Count'
        ]

        scaled_data, scaled_data_validation = self.scale_columns(
            scaling_columns=scaling_labels,
            train_data=self.train_data,
            validation_data=self.validation_data,
        )

        one_hot_encoded_data, one_hot_encoded_validation = self.convert_one_hot_labels(
            categorical_labels=one_hot_labels,
            train_data=self.train_data,
            validation_data=self.validation_data
        )
        categorical_data, categorical_validation = self.convert_categorical_labels(
            categorical_labels=categorical_labels,
            train_data=self.train_data,
            validation_data=self.validation_data
        )

        self.train_data = pd.concat([
            scaled_data,
            one_hot_encoded_data,
            categorical_data,
        ], axis=1)
        self.validation_data = pd.concat([
            scaled_data_validation,
            one_hot_encoded_validation,
            categorical_validation,
        ], axis=1)

        print(
            "Jigsaw dataset {0} successfully loaded.".format(
                self.train_data.shape,
            )
        )

