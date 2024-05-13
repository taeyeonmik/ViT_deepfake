import os
import math
import yaml
import argparse
import collections
from progress.bar import ChargingBar

import numpy as np
from PIL import Image
import torch
from torch.optim import lr_scheduler

from dataset import DeepFakesDataset
from efficient_vit import EfficientViT
from utils import getDataset, check_correct, shuffle_dataset, create_label, get_n_params


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.') #10
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--models', default='efficientvit', type=str, metavar='PATH',
                        help='Select a model architecture among [efficientvit, efficientnet, vit]')
    parser.add_argument('--datapath', type=str, default='./data/',
                        help="Path of image for training")
    parser.add_argument('--config', default='./efficient-vit/configs/configuration.yaml', type=str,
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0,
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5,
                        help="How many epochs wait before stopping for validation loss not improving.")
    opt = parser.parse_args()
    print(opt)

    # dataset
    DATAPATH = opt.datapath
    # VALIDPATH = opt.validpath
    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    # trainset, validset, testset = get_data_pair(DATAPATH)
    tr, vl, ts = getDataset(DATAPATH)
    trainset = tr['ffhq'] + tr['stylegan_ffhq']
    validset = vl['ffhq'] + vl['stylegan_ffhq']
    testset = ts['ffhq'] + ts['stylegan_ffhq']

    trainset = shuffle_dataset(trainset)
    validset = shuffle_dataset(validset)
    testset = shuffle_dataset(testset)
    # create labels
    trlb = create_label(trainset, DFcheckStr='F_SyFQ')
    vllb = create_label(validset, DFcheckStr='F_SyFQ')
    tslb = create_label(testset, DFcheckStr='F_SyFQ')

    trainset, validset, testset = (trainset, trlb), (validset, vllb), (testset, tslb)

    train_samples = len(trainset[0])
    validation_samples = len(validset[0])
    test_samples = len(testset[0])

    # Print some statistics
    print("Train images:", train_samples, "Validation images:", validation_samples)
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image for image in trainset[1])
    print(train_counters)

    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image for image in validset[1])
    print(val_counters)

    print("__TEST STATS__")
    test_counters = collections.Counter(image for image in testset[1])
    print(test_counters)
    print("___________________")


    # model
    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    if opt.models == 'efficientvit':
        model = EfficientViT(config=config, channels=channels, selected_efficient_net=opt.efficient_net)
    elif opt.models == 'efficientnet':
        model = EfficientViT(config=config, channels=channels, selected_efficient_net=opt.efficient_net, vit=False)
    elif opt.models == 'vit':
        model = EfficientViT(config=config, channels=channels, selected_efficient_net=opt.efficient_net, effnet=False)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'],
                                weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'],
                                    gamma=config['training']['gamma'])
    # resume
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[
                                 0]) + 1  # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    # Create the data loaders
    train_labels = trainset[1]
    trainset = DeepFakesDataset(trainset[0], trainset[1], config['model']['image-size'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                     batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                     pin_memory=False, drop_last=False, timeout=0,
                                     worker_init_fn=None, prefetch_factor=2,
                                     persistent_workers=False)
    del trainset

    valid_labels = validset[1]
    validset = DeepFakesDataset(validset[0], validset[1], config['model']['image-size'], mode='validation')
    validloader = torch.utils.data.DataLoader(validset, batch_size=config['training']['bs'], shuffle=True,
                                         sampler=None,
                                         batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                         pin_memory=False, drop_last=False, timeout=0,
                                         worker_init_fn=None, prefetch_factor=2,
                                         persistent_workers=False)
    del validset

    # train
    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf

    import time
    tr_loss = []
    vl_loss = []
    for t in range(starting_epoch, opt.num_epochs):
        if not_improved_loss == 5:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0

        bar = ChargingBar('EPOCH #' + str(t), max=(len(trainloader) * config['training']['bs']) + len(validloader))
        train_correct = 0
        positive = 0
        negative = 0
        gtp, gfp, gtn, gfn = 0, 0, 0, 0

        cur_step = 0
        begin = time.time()
        for index, (images, labels) in enumerate(trainloader):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()

            y_pred = model(images)
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels.float())

            corrects, positive_class, negative_class, tpfptnfn = check_correct(y_pred, labels)
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            gtp += tpfptnfn[0]
            gfp += tpfptnfn[1]
            gtn += tpfptnfn[2]
            gfn += tpfptnfn[3]
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)

            if index % 25 == 0:  # Intermediate metrics print
                # interm = time.time()
                print(f"[{cur_step+1}/{25}]", "Loss: ", total_loss / counter, "Accuracy: ",
                      train_correct / (counter * config['training']['bs']), "Train 0s: ", negative, "Train 1s:",
                      positive)
                cur_step += 1
            for i in range(config['training']['bs']):
                bar.next()
        later = time.time()
        difference = int(later - begin)
        print(f"#{t+1}/{opt.num_epochs} training time: {difference}s")
        cur_step = 0

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        vgtp, vgfp, vgtn, vgfn = 0, 0, 0, 0
        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(validloader):
            val_images = np.transpose(val_images, (0, 3, 1, 2))

            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            val_pred = val_pred.cpu()
            val_loss = loss_fn(val_pred, val_labels.float())
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class, tpfptnfn = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class
            vgtp += tpfptnfn[0]
            vgfp += tpfptnfn[1]
            vgtn += tpfptnfn[2]
            vgfn += tpfptnfn[3]
            bar.next()

        scheduler.step()
        bar.finish()

        total_val_loss /= val_counter
        val_correct /= validation_samples
        val_precision = vgtp / (vgtp + vgfp)
        val_recall = vgtp / (vgtp + vgfn)
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        previous_loss = total_val_loss
        print("#" + str(t+1) + "/" + str(30) + " loss:" +
                  str(total_loss) + " accuracy:" + str(train_correct) + " val_loss:" + str(
                total_val_loss) + " val_accuracy:" + str(val_correct) + " val_precision:" + str(val_precision) +
                " val_recall:" + str(val_recall) + " val_f1:" + str(val_f1) + " val_0s:" + str(val_negative) + "/" + str(
                np.count_nonzero(valid_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(
                np.count_nonzero(valid_labels == 1)) + "\n")

        tr_loss.append(total_loss)
        vl_loss.append(total_val_loss)

        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint')
        torch.save(model.state_dict(), os.path.join('./checkpoint',
                                                    "efficientnetB" + str(0) + "_checkpoint" + str(
                                                        t) + "_" + "test"))
    print(tr_loss)
    print(vl_loss)