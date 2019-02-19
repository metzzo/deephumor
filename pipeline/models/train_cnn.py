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
    model.to(device)

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

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            evaluations[phase].reset()

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.
            for _, inputs, _, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = model.get_predictions(outputs=outputs)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                evaluations[phase].add_entry(predictions=preds, actual_label=labels, loss=loss.item())

            print('{0} evaluation:\n {1}'.format(
                phase, str(evaluations[phase])))

            # deep copy the model
            # TODO
            #if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model