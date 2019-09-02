import datetime
import os
import time
import copy

import torch
from torch.utils.data import DataLoader

from evaluation.overall_evaluation import OverallEvaluation

import pandas as pd

def train_cnn_model(
        model,
        criterion,
        optimizer,
        scheduler,
        training_dataset,
        validation_dataset,
        test_dataset,
        batch_size,
        device,
        num_epochs=25):
    import time

    start = time.time()

    network = model.network
    network.to(device)
    #LOAD_MODEL = '/home/rfischer/Documents/DeepHumor/deephumor/final_models/transfer_learning.pth'
    LOAD_MODEL = False

    optimizer = optimizer(params=model.optimization_parameters)
    scheduler = scheduler(optimizer=optimizer)

    train_dl = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders = {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
    }

    evaluations = {
        "train": OverallEvaluation(
            num=len(dataloaders["train"]), batch_size=batch_size
        ).add_evaluations(model.train_evaluations),
        "val": OverallEvaluation(
            num=len(dataloaders["val"]), batch_size=batch_size
        ).add_evaluations(model.validation_evaluations),
    }
    target_path = os.path.abspath("../../models/{0}_cnn_model.pth".format(
        str(datetime.datetime.today()).replace('-', '').replace(':', '').replace(' ', '_')
    ))

    if not LOAD_MODEL:
        since = time.time()

        best_network_wts = copy.deepcopy(network.state_dict())
        best_acc = 0.0

        patience = 0

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
                print("New Best!")
                best_acc = epoch_acc
                best_network_wts = copy.deepcopy(network.state_dict())
                torch.save(network.state_dict(), target_path)
                patience = 0
            else:
                patience += 1

            if patience > 4:
                print("ran out of patience")
                break

            print()

        end = time.time()
        print("Time of training ", end - start)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        network.load_state_dict(best_network_wts)
    else:
        #network.load_state_dict({"state_dict": torch.load(LOAD_MODEL)})
        network.load_state_dict(torch.load(LOAD_MODEL))

    # do test evaluation
    network.eval()  # Set network to evaluate mode
    for phase in ['val', 'test']:
        evaluations['val'].reset()
        # Iterate over data.

        all_predictions = []
        all_labels = []
        for data in dataloaders[phase]:
            inputs, labels = model.get_input_and_label(data)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = network(inputs)
            labels = model.get_labels(labels=labels)
            preds = model.get_predictions(outputs=outputs)
            #print(inputs)

            # statistics
            evaluations['val'].add_entry(predictions=preds, actual_label=labels, loss=42)

            all_predictions += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())

        df = pd.DataFrame(data={
            "predictions": all_predictions,
            "labels": all_labels,
        })
        df.to_csv('{}_{}.csv'.format(target_path, phase), sep=';')

        print('{0} evaluation:\n {1}'.format(
            phase, str(evaluations['val'])))

    return network
