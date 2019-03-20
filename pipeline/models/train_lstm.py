import datetime
import os
import string
import time
import copy

import torch
from torch.utils.data import DataLoader

from evaluation.overall_evaluation import OverallEvaluation

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_tensor(letter)] = 1
    return tensor


def train_lstm_model(
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
        "train": OverallEvaluation(num=len(dataloaders["train"]), ignore_loss=False, batch_size=batch_size),
        "val": OverallEvaluation(num=len(dataloaders["val"]), ignore_loss=True, batch_size=batch_size),
    }

    target_path = os.path.abspath("./../models/{0}_lstm_model.pth".format(
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
            for _, inputs, _, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = network(inputs)
                    preds = model.get_predictions(outputs=outputs)
                    labels = model.get_labels(labels=labels)

                    loss = criterion(outputs, labels)

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
