from __future__ import absolute_import, division, print_function, unicode_literals

import architecture
import argparse
# import cifar
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from utils.trainutil import (
    train_directory_setup,
    train_log_results,
    train,
    valid_highdim,
    valid_category,
    valid_lowdim,
    test_highdim,
    test_category,
    test_lowdim,
    test_highdim_dist
)
# from torchtoolbox.nn import LabelSmoothingLoss
from sklearn.manifold import TSNE
from digit import *

if __name__ == "__main__":
    # Params setup
    parser = argparse.ArgumentParser(description="CIFAR High-dimensional Model.")
    parser.add_argument("--epochs", type=int, default=200, help='batch size')
    parser.add_argument("--batch_size", type=int, default=128, help='batch size')
    parser.add_argument("--data_frac", type=float, default=1.0, help='fraction of training data used')
    parser.add_argument(
        "--label",
        type=str,
        help="Label in [speech, uniform, shuffle, composite, random, uniform, lowdim, bert, glove]",
    )
    parser.add_argument(
        "--model", type=str, help="Image encoder in [vgg16, vgg19, resnet20, resnet110, resnet32]"
    )
    parser.add_argument("--seed", type=int, help="Manual seed.", required=True)
    parser.add_argument("--level", type=int, default=100, help="Data level.")
    parser.add_argument(
        "--label_dir",
        type=str,
        help="Directory where labels are stored",
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where CIFAR datasets are stored",
        default="./data",
    )
    parser.add_argument(
        "--base_dir", type=str, default="./outputs", help="Directory to save outputs"
    )
    parser.add_argument("--dataset", type=str, help="Dataset to train on")
    parser.add_argument(
        "--smoothing", type=float, default=0, help="Label smoothing level (default: 0)."
    )

    args = parser.parse_args()
    label = args.label
    data_dir = args.data_dir
    model_name = args.model
    seq_seed = args.seed
    # data_level = args.level
    base_dir = args.base_dir
    label_dir = args.label_dir
    dataset = args.dataset
    smoothing = args.smoothing
    batch_size = args.batch_size
    epoch_stop = args.epochs
    data_frac = args.data_frac
    data_level = data_frac

    assert dataset in ("s2m", "u2m", "m2u")

    less_data = data_level < 100
    assert label in (
        "speech",
        "uniform",
        "shuffle",
        "composite",
        "random",
        "bert",
        "lowdim",
        "glove",
        "category",
        "constant"
    )

    if smoothing > 0:
        label = "category_smooth{}".format(smoothing)

    assert model_name in ("vgg16", "vgg19", "resnet20", "resnet110", "resnet32")
    if less_data:
        assert data_level < 90
    print(
        "Start training {}% {} {} model with manual seed {} and model {}.".format(
            data_level, dataset, label, seq_seed, model_name
        )
    )

    # Directory setup
    (
        best_model_path,
        checkpoint_path,
        log_path,
        snapshots_folder,
    ) = train_directory_setup(
        label, model_name, dataset, seq_seed, data_level, base_dir
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # Loads train, validation, and test data

    # num_classes = int(dataset.split("cifar")[-1])
    # trainloader = cifar.get_train_loader(
    #     data_dir, label, num_classes, num_workers, 128, seq_seed, data_level, label_dir
    # )
    # validloader = cifar.get_valid_loader(
    #     data_dir, label, num_classes, num_workers, 100, seq_seed, label_dir
    # )
    # testloader = cifar.get_test_loader(
    #     data_dir, label, num_classes, num_workers, 100, label_dir
    # )

    num_classes = 10
    loaders, datasets = digit_load(batch_size, dataset, label_dir, data_frac)
    trainloader = loaders["source_tr"]
    validloader = loaders["source_te"]
    testloader  = loaders["test"]

    # Model setup
    if "category" in label or label in ("lowdim", "glove"):
        if label == "glove":
            model = architecture.CategoryModel(model_name, 50)
        else:
            model = architecture.CategoryModel(model_name, num_classes)
    elif label == "bert":
        model = architecture.BERTHighDimensionalModel(model_name, num_classes)
    else:
        model = architecture.HighDimensionalModel(model_name, num_classes)

    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[49, 99, 149], gamma=0.1
    )
    # epoch_stop = 600

    if "category" in label:
        if smoothing > 0:
            criterion = LabelSmoothingLoss(num_classes, smoothing=smoothing)
        else:
            criterion = nn.CrossEntropyLoss()

    else:
        criterion = nn.SmoothL1Loss()

    # Initializes training
    load_from_checkpoint = False
    if load_from_checkpoint:
        checkpoint = torch.load(checkpoint_path)
        epoch_start = checkpoint["epoch"]
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        valid_acc = checkpoint["valid_acc"]
        model.load_state_dict(checkpoint["model_state_dict"])
        model = nn.DataParallel(model.module).to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        min_valid_loss = np.min(valid_loss)
        max_valid_acc = np.max(valid_acc)
        print(
            "Loaded checkpoint from epoch {} with min valid loss {} | max valid acc {}".format(
                epoch_start, min_valid_loss, max_valid_acc
            )
        )
    else:
        epoch_start = 0
        min_valid_loss = float("inf")
        max_valid_acc = 0.0
        train_loss = []
        valid_loss = []
        valid_acc = []

    # Trains model from epoch_start to epoch_stop
    for epoch in range(epoch_start, epoch_stop):
        new_train_loss = train(model, trainloader, optimizer, criterion, device)
        if "category" in label:
            new_valid_loss, new_valid_acc = valid_category(
                model, validloader, criterion, device
            )
        elif label in ("lowdim", "glove"):
            new_valid_loss, new_valid_acc = valid_lowdim(
                model, validloader, criterion, device
            )
        else:
            new_valid_loss, new_valid_acc = valid_highdim(
                model, validloader, criterion, device
            )

        scheduler.step(epoch)
        train_loss.append(new_train_loss)
        valid_loss.append(new_valid_loss)
        valid_acc.append(new_valid_acc)
        print(
            "Epoch {} train loss {} | valid loss {} | valid acc {}".format(
                epoch + 1, new_train_loss, new_valid_loss, new_valid_acc
            )
        )
        if new_valid_acc > max_valid_acc or (
            new_valid_acc == max_valid_acc and new_valid_loss < min_valid_loss
        ):
            print("Saving new best checkpoint...")
            min_valid_loss = new_valid_loss
            max_valid_acc = new_valid_acc
            torch.save(model.state_dict(), best_model_path)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            },
            checkpoint_path,
        )
        if epoch % 50 == 49:
            snapshot_file = "{}_{}_seed{}_{}_epoch{}_model.pth".format(
                data_frac, label, seq_seed, model_name, epoch + 1
            )
            snapshot_path = os.path.join(snapshots_folder, snapshot_file)
            torch.save(model.state_dict(), snapshot_path)

            # save features 
            feat = []
            targ = []
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    feature = model.get_feature(inputs)
                    feat.append(feature)
                    targ.append(targets)
            feat = torch.cat(feat).cpu().numpy()
            targ = torch.cat(targ).cpu().numpy()

            tsne = TSNE()
            feat_2d = tsne.fit_transform(feat)
            np.savez(os.path.join(snapshots_folder, 'test_' + snapshot_file.replace('.pth', '.npz')), feat=feat_2d, targ=targ)
            print(f'snap shot saved at {snapshot_path}')

    # Evaluates the best model
    model.load_state_dict(
        torch.load(best_model_path, map_location=torch.device(device))
    )

    # Test model
    if "category" in label:
        test_loss, test_acc = test_category(model, testloader, criterion, device)
    elif label in ("lowdim", "glove"):
        test_loss, test_acc = test_lowdim(model, testloader, criterion, device)
    else:
        # test_loss, test_acc = test_highdim(model, testloader, criterion, device)
        test_loss, test_acc = test_highdim_dist(model, testloader, device)
    print(
        "Label {}: seed {}, model {}, test loss {}, test acc {}".format(
            label, seq_seed, model_name, test_loss, test_acc
        )
    )

    # Logs results
    train_log_results(log_path, model_name, data_level, seq_seed, test_loss, test_acc)
