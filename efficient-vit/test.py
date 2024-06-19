import yaml
import argparse
import os
import numpy as np
import collections
import torch
from progress.bar import Bar
from multiprocessing import Manager

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score

from dataset import DeepFakesDataset
from utils import getDataset, create_label
from efficient_vit import EfficientViT

def draw_roc_curves(correct_labels, preds, model_name):
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')

    fpr, tpr, th = metrics.roc_curve(correct_labels, preds)

    auc_model = metrics.auc(fpr, tpr)

    youden_index = tpr - fpr
    optimal_threshold_index = np.argmax(youden_index)
    optimal_threshold = th[optimal_threshold_index]
    print(f'Threshold optimal: {optimal_threshold:.5f}')

    plt.plot(fpr, tpr, label="Model_" + model_name + ' (area = {:.3f})'.format(auc_model))
    plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', label='Optimal threshold')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, th, auc_model


# Main body
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', default=2, type=int, help='Number of data loader workers.')
    parser.add_argument('--models', default='efficientvit', type=str, metavar='PATH', help='Select a model architecture among [efficientvit, efficientnet]')
    parser.add_argument('--model_path', default='', type=str, metavar='PATH', help='Path to model checkpoint (default: none).')
    parser.add_argument('--datapath', type=str, default='/Users/taeyeon/PersonalProjects/ViT_deepfake/', help="Path of image for training")
    parser.add_argument('--config', default='./efficient-vit/configs/configuration.yaml', type=str,help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, help="Which EfficientNet version to use (0 or 1, default: 0)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument('--celeba', type=int, default=0, help="1 if you want to do the inference with StyleGAN_CelebA dataset.")

    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)

    # dataset
    DATAPATH = opt.datapath
    print(DATAPATH)
    _, _, ts = getDataset(DATAPATH)

    if not opt.celeba:
        testset = ts['ffhq'] + ts['stylegan_ffhq']
    else:
        testset = ts['stylegan_celeba']

    # create labels
    tslb = create_label(testset, DFcheckStr='F_Sy')

    testset = (testset, tslb)
    test_samples = len(testset[0])

    # Print some statistics
    print("Test images:", test_samples)
    print("__TEST STATS__")
    test_counters = collections.Counter(image for image in testset[1])
    print(test_counters)
    class_weights = test_counters[0] / test_counters[1]
    print("Weights", class_weights)
    print("___________________")

    # apply transforms
    testset = DeepFakesDataset(testset[0], testset[1], config['model']['image-size'], mode='validation')
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, sampler=None,
                                              batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                              pin_memory=False, drop_last=False, timeout=0,
                                              worker_init_fn=None, prefetch_factor=2,
                                              persistent_workers=False)
    del testset

    # define an efficientnet type
    if (opt.efficient_net == 0) or (opt.efficient_net == 1):
        channels = 1280
    else:
        channels = 2560

    if opt.models:
        if opt.models == 'efficientvit':
            model = EfficientViT(config=config, channels=channels, selected_efficient_net=opt.efficient_net)
        elif opt.models == 'efficientnet':
            model = EfficientViT(config=config, channels=channels, selected_efficient_net=opt.efficient_net, vit=False)
        # load state dict
        if os.path.exists(opt.model_path):
            model.load_state_dict(torch.load(opt.model_path))
            print("state dict loaded.")
            model.eval()
            model = model.cuda()
        else:
            print("No model found.")
            exit()
    else:
        print("Model architecture not defined.")
        exit()

    model_name = os.path.basename(opt.model_path)

    OUTPUT_DIR = config['inference']['infer-path']
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    mgr = Manager()

    correct_test_labels = np.asarray(tslb)
    preds = []
    bar = Bar('Predicting', max=len(testset[0]))

    import time
    begin = time.time()
    with torch.no_grad():
        for index, (images, labels) in enumerate(testloader):
            images = np.transpose(images, (0, 3, 1, 2))
            images = images.cuda()
            labels = labels.unsqueeze(1)

            pred = model(images)
            pred = pred.cpu()
            pred = [np.asarray(torch.sigmoid(p).detach()) for p in pred]
            preds.extend(pred)

            bar.next()
    later = time.time()
    difference = int(later - begin)
    print(f"#inference time: {difference}s")
    bar.finish()

    # loss, accuracy, f1, auc calcul
    loss_fn = torch.nn.BCEWithLogitsLoss()
    tensor_labels = torch.tensor([[float(label)] for label in correct_test_labels])
    tensor_preds = torch.tensor(preds)

    loss = loss_fn(tensor_preds, tensor_labels).numpy()
    accuracy = accuracy_score(np.asarray(preds).round(), correct_test_labels)

    f1 = f1_score(correct_test_labels, np.asarray(preds).round())
    print(model_name, "Loss:", loss, "Test Accuracy:", accuracy, "F1", f1)

    fpr, tpr, th, auc = draw_roc_curves(correct_test_labels, preds, model_name)