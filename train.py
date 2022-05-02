import os
import numpy as np
import torch
from PIL import Image
import torchvision
import utils as ut
import torch.utils.data 
from dicto import dict_Gordon
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from drinks_dataset import ImageDataSet
import transforms as T
from engine import train_one_epoch, evaluate

def trainer():

    def get_transform(train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    #dataset = ImageDataSet('./drinks','drinks/labels_train.csv', get_transform(train=True))
    #dataset_test = ImageDataSet('./drinks','drinks/labels_test.csv',get_transform(train=False))

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    
    num_classes = 4
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    dataset = ImageDataSet('./drinks', 'drinks/labels_train.csv', transforms = T.Compose([T.ToTensor()]))
    dataset_test = ImageDataSet('./drinks', 'drinks/labels_test.csv', transforms = T.Compose([T.ToTensor()]))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=ut.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=ut.collate_fn)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        print("training")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

trainer()