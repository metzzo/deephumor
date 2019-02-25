import datetime
import os
import time
import copy

import torch
from torch.utils.data import DataLoader

from evaluation.overall_evaluation import OverallEvaluation


def train_cnn_model(
        model,
        criterion,
        optimizer,
        scheduler,
        training_dataset,
        validation_dataset,
        batch_size,
        device,
        num_epochs=25):
    network = model.network
    network.to(device)

    optimizer = optimizer(params=model.optimization_parameters)
    scheduler = scheduler(optimizer=optimizer)

    dataloaders = {
        "train": DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
    }

    evaluations = {
        "train": OverallEvaluation(
            num=len(dataloaders["train"]), batch_size=batch_size
        ).add_evaluations(model.train_evaluations),
        "val": OverallEvaluation(
            num=len(dataloaders["val"]), batch_size=batch_size
        ).add_evaluations(model.validation_evaluations),
    }

    target_path = os.path.abspath("./../models/{0}_cnn_model.pth".format(
        str(datetime.datetime.today()).replace('-', '').replace(':', '').replace(' ', '_')
    ))
    since = time.time()

    best_network_wts = copy.deepcopy(network.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            evaluations[phase].reset()

            if phase == 'train':
                scheduler.step()
                network.train()  # Set network to training mode
            else:
                network.eval()   # Set network to evaluate mode

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs, labels = model.get_input_and_label(data)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = network(inputs)
                    labels = model.get_labels(labels=labels)

                    loss = criterion(outputs, labels)

                    preds = model.get_predictions(outputs=outputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                evaluations[phase].add_entry(predictions=preds, actual_label=labels, loss=loss.item())

            print('{0} evaluation:\n {1}'.format(
                phase, str(evaluations[phase])))

        # deep copy the network
        epoch_acc = evaluations['val'].accuracy_evaluation.accuracy
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_network_wts = copy.deepcopy(network.state_dict())
            torch.save(network.state_dict(), target_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    network.load_state_dict(best_network_wts)

    return network
