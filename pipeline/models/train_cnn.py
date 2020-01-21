import datetime
import shutil

import matplotlib
import os
import time
import copy

import torch
from PIL import Image, ImageFont
from numpy import random
from torch.utils.data import DataLoader

from confusion_matrix import plot_confusion_matrix_from_data
from evaluation.overall_evaluation import OverallEvaluation

import pandas as pd

from random_seed import set_random_seed


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
    LOAD_MODEL = '/home/rfischer/Documents/DeepHumor/deephumor/final_models/transfer_learning.pth'
    #LOAD_MODEL = False

    optimizer = optimizer(params=model.optimization_parameters)
    scheduler = scheduler(optimizer=optimizer)

    train_dl = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dl = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dataloaders = {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
    }
    datasets = {
        "val": validation_dataset,
        "test": test_dataset,
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
    set_random_seed(42)


    # do test evaluation
    network.eval()  # Set network to evaluate mode

    DETAIL_DIR = '../../detail/'
    shutil.rmtree(DETAIL_DIR)
    os.mkdir(DETAIL_DIR)
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    font = ImageFont.truetype("../../font.ttf", 24)

    for phase in ['val', 'test']:
        relevant_activations = {}
        evaluations['val'].reset()
        # Iterate over data.

        all_predictions = []
        all_labels = []

        #f, axes = plt.subplots(7, 2, figsize=(128, 128), sharex=True)

        import matplotlib.image as image
        for data in dataloaders[phase]:
            inputs, labels = model.get_input_and_label(data)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = network(inputs)
            if len(relevant_activations) < 7:
                if phase == 'test':
                    cid, _, _ = data
                    for j in range(inputs.size(0) - 1):
                        label = int(labels[j].cpu().detach().numpy())
                        if label not in relevant_activations:
                            output = torch.nn.functional.softmax(outputs[j]).cpu().detach().numpy()
                            cur_id = int(cid[j])

                            output = pd.DataFrame(data={
                                key: output[key] for key in range(1, 8)
                            }, index=['data']).T
                            output.reset_index(level=0, inplace=True)
                            output.rename(columns={"index": "Funniness", "data": "Probability"}, inplace=True)
                            matplotlib.pyplot.xlim(0.0, 1.0)

                            title = '{} for Image {}'.format('Validation' if phase == 'val' else 'Test', label)
                            clean_title = title.replace(' ', '_')
                            f = {'family': 'serif', 'serif': ['computer modern roman']}
                            plt.rc('font', **f)
                            #sns.set(rc={'figure.figsize': (11.7, 8.27)})
                            with sns.plotting_context(font_scale=1.5):
                                ax = sns.barplot(
                                    x='Probability',
                                    y='Funniness',
                                    data=output,
                                    color="b",
                                    ci=None,
                                    orient='h',
                                    dodge=False

                                )
                                ax.set_title('True Funniness {}'.format(label), fontsize=18)
                                ax.set_xlabel('Probability', fontsize=18)
                                ax.set_ylabel('Funniness', fontsize=18)
                                ax.tick_params(labelsize=16)

                                ds = datasets[phase]
                                row = ds.df.iloc[cur_id]
                                filename = os.path.join(ds.root_dir, row['filename'])
                                punchline = row['punchline'].replace('\\n', ' ')
                                print("punchline", punchline)
                                #plt.show()
                                plt.tight_layout()
                                plt.savefig(os.path.join(DETAIL_DIR, clean_title + '.png'))

                            img = Image.open(filename)
                            draw = ImageDraw.Draw(img)

                            import textwrap
                            lines = textwrap.wrap(punchline, width=40)

                            w = img.width
                            h = img.height - len(lines) * font.getsize('|')[1] - 10

                            draw.rectangle(((0, h - 10), (w, img.height + 10)), fill='white')

                            y_text = h
                            for line in lines:
                                width, height = font.getsize(line)
                                draw.text(((w - width) / 2, y_text), line, font=font)
                                y_text += height

                            img.save(os.path.join(DETAIL_DIR, clean_title + '_cartoon.jpg'))

                            #shutil.copyfile(filename, os.path.join(DETAIL_DIR, clean_title + '_cartoon.jpg'))

                            #img = np.asarray(Image.open(filename).convert("L"))
                            #img = image.imread(filename)
                            #axes[label - 1, 0].axis('off')
                            #plt.figure(figsize=(20, 2))
                            #axes[label - 1, 0].imshow(random.rand(8, 90), cmap='gray', aspect='auto', interpolation='nearest')

                            #print(label, " ", output, " ", cur_id)

                            relevant_activations[label] = 42
            else:
                break

            labels = model.get_labels(labels=labels)
            preds = model.get_predictions(outputs=outputs)
            #print(inputs)

            # statistics
            evaluations['val'].add_entry(predictions=preds, actual_label=labels, loss=42)

            all_predictions += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())
        plt.show()

        df = pd.DataFrame(data={
            "predictions": all_predictions,
            "labels": all_labels,
        })
        df.to_csv('{}_{}.csv'.format(target_path, phase), sep=';')

        print('{0} evaluation:\n {1}'.format(
            phase, str(evaluations['val'])))
        """
        plot_confusion_matrix_from_data(
            y_test=all_labels,
            predictions=all_predictions,
            columns=[
                'funniness 1',
                'funniness 2',
                'funniness 3',
                'funniness 4',
                'funniness 5',
                'funniness 6',
                'funniness 7',
            ],
            title='{} confusion matrix'.format('Validation' if phase=='val' else 'Test')
        )
        """

    return network
