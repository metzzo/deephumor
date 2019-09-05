"""
This file is used to transform the training set to a CSV file.
This allows analysis in Excel.
"""
import argparse
import pickle

import torch
from torch.utils.data import DataLoader

from cnn_experiments.factory import get_model
from datamanagement.subset import Subset
from datamanagement.factory import get_subset
from evaluation.overall_evaluation import OverallEvaluation


def setup_predict_cnn(parser: argparse.ArgumentParser, group):
    group.add_argument('--predict_cnn', action="store_true")

    parser.add_argument('--configuration', type=str)

    def predict_cnn(args, device):
        if not args.predict_cnn:
            return

        selected_model = get_model(model_name=args.model)
        network = selected_model.network
        network.to(device)

        validation_ds = selected_model.Dataset(
            file_path=get_subset(dataset_path=args.source, subset=Subset.VALIDATION),
            model=selected_model,
            trafo=selected_model.get_validation_transformation(),
        )

        dataloader = DataLoader(
            dataset=validation_ds,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        evaluation = OverallEvaluation(
            num=len(dataloader),
            batch_size=32
        ).add_evaluations(selected_model.validation_evaluations)

        evaluation.reset()
        network.load_state_dict(torch.load(args.configuration))
        network.eval()

        # Iterate over data.
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = selected_model.get_input_and_label(data)
                inputs = inputs.to(device)
                labels = labels.to(device)


                outputs = network(inputs)
                labels = selected_model.get_labels(labels=labels)
                preds = selected_model.get_predictions(outputs=outputs)
                _, top_five = outputs.topk(5, 1, True, True)

                # statistics
                evaluation.add_entry(predictions=preds, actual_label=labels, loss=None, top_five=top_five)

        print('Prediction Evaluation:\n {0}'.format(str(evaluation)))



    return predict_cnn
