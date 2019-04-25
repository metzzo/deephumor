import copy
import datetime
import os
import time

from torch.utils.data import DataLoader

from classifier.base import BaseClassifier
from evaluations.performance_evaluation import PerformanceEvaluation
from evaluations.loss_evaluation import LossEvaluation
from evaluations.overall_evaluation import OverallEvaluation
from torch.optim import lr_scheduler

import torch
import torch.nn as nn

from utility.test_collector import TestCollector


class NeuralNetworkClassifierMixin(BaseClassifier):
    @property
    def network_class(self):
        raise NotImplementedError()

    def __init__(self, args):
        super(NeuralNetworkClassifierMixin, self).__init__(args)

        if not hasattr(self, 'dataset'):
            self.dataset = None
            raise RuntimeError("Missing dataset")

        if not hasattr(self, 'args'):
            self.args = None
            raise RuntimeError("Missing args")

        use_cuda = torch.cuda.is_available() and not args.use_cpu
        if args.n_jobs > 0:
            torch.set_num_threads(args.n_jobs)

        print("Uses CUDA: {0}".format(use_cuda))
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.network = self.network_class(
            num_classes=self.dataset.num_classes,
            num_input=self.dataset.num_input
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.network.to(self.device)

        optimizer = torch.optim.SGD(
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
            nesterov=True,
            params=self.network.parameters(),
        )
        scheduler = lr_scheduler.StepLR(
            step_size=self.args.steplr_step_size,
            gamma=self.args.steplr_gamma,
            optimizer=optimizer
        )

        batch_size = self.args.batch_size
        num_epochs = self.args.num_epochs

        dataloaders = {
            "train": DataLoader(
                dataset=self.dataset.pytorch_dataset(usage='train'),
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.args.n_jobs - 1
            ),
            "val": DataLoader(
                dataset=self.dataset.pytorch_dataset(usage='val'),
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.args.n_jobs - 1
            ),
        }

        evaluations = {
            "train": OverallEvaluation(
                num=len(dataloaders["train"]), experiment=self
            ).add_evaluations([
                LossEvaluation,
                PerformanceEvaluation
            ]),
            "val": OverallEvaluation(
                num=len(dataloaders["val"]), experiment=self
            ).add_evaluations([
                PerformanceEvaluation
            ]),
        }

        target_path = os.path.abspath(
            os.path.join(
                self.args.directory,
                "models",
                "{0}_cnn_model.pth".format(
                    str(datetime.datetime.today()).replace('-', '').replace(':', '').replace(' ', '_')
                )
            )
        )
        since = time.time()

        best_network_wts = copy.deepcopy(self.network.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                evaluations[phase].reset()

                if phase == 'train':
                    scheduler.step()
                    self.network.train()  # Set network to training mode
                else:
                    self.network.eval()  # Set network to evaluate mode

                # Iterate over data.
                for data in dataloaders[phase]:
                    inputs, labels = data

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.network(inputs)
                        loss = self.criterion(outputs, labels)
                        preds = torch.max(outputs, 1)[1]

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    evaluations[phase].add_entry(
                        predictions=preds.cpu().numpy(),
                        actual_label=labels.cpu().numpy(),
                        loss=loss.item()
                    )

                print('{0} evaluation:\n {1}'.format(
                    phase, str(evaluations[phase])))

            # deep copy the network
            epoch_acc = evaluations['val'].accuracy_evaluation.accuracy
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_network_wts = copy.deepcopy(self.network.state_dict())
                #torch.save(self.network.state_dict(), target_path)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.network.load_state_dict(best_network_wts)

    def test(self):
        self.network.to(self.device)

        dataloader = DataLoader(
            dataset=self.dataset.pytorch_dataset(usage='test'),
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
        collector = TestCollector(dataset=self.dataset)
        self.network.eval()

        # Iterate over data.
        with torch.no_grad():
            for data in dataloader:
                inputs, ids = data
                inputs = inputs.to(self.device)

                outputs = self.network(inputs)
                preds = torch.max(outputs, 1)[1]

                # statistics
                collector.collect_test(
                    id=ids.cpu().numpy(),
                    predicted_class=preds.cpu().numpy(),
                )
        collector.save_csv(
            path=os.path.join(self.args.directory, 'test_result.csv')
        )
