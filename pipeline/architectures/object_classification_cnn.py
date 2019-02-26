from functools import partial

from torchvision import transforms

from architectures.tuberlin_classification_cnn import TUBerlinClassificationModel
from datamanagement.object_dataset import ObjectDataset
from datamanagement.tuberlin_dataset import TUBerlinDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation
from evaluation.detailed_evaluation import DetailedEvaluation


class ObjectClassificationModel(TUBerlinClassificationModel):
    @property
    def Dataset(self):
        return ObjectDataset

    def get_validation_transformation(self):
        return [
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
        ]

    @property
    def validation_evaluations(self):
        return super(ObjectClassificationModel, self).validation_evaluations + [partial(DetailedEvaluation, model=self)]

    def get_human_readable_class(self, cl, is_predicted):
        if is_predicted:
            return TUBerlinDataset.Classes[cl]
        else:
            return ObjectDataset.Classes[cl]
