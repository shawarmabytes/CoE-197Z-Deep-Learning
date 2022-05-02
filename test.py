#import os
#import numpy as np
import torch
#from PIL import Image
import utils as ut
import torch.utils.data 
#from dicto import dict_Gordon
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from drinks_dataset import ImageDataSet
import transforms as T
from engine import evaluate

def tester():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 4
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    dataset_test = ImageDataSet('./drinks', 'drinks/labels_test.csv', transforms = T.Compose([T.ToTensor()]))


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=ut.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    drinks = torch.load("drinks_dataset_trained_model.pth")
    model.load_state_dict(drinks["state_dict"])
    model.to(device)
    model.eval()
    evaluate(model, data_loader_test, device=device)

tester()
