# CoE-197Z-Deep-Learning-Object-Detection

## About the project
This project used ***Faster R-CNN MobileNetV3-Large 320 FPN*** as its pre-trained model and fine-tuned using the drinks dataset, which can be found on this repository's release or accessed through this [google drive link](https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view?usp=sharing). 

For more information about torchvision and object detection, click [here](https://github.com/pytorch/vision/tree/main/references/detection)


## Install requirements
```
pip install -r requirements.txt
```
Note: The requirements.txt file assumes that you have torch and torchvision installed with cuda enabled.

Otherwise, you can install them by running:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

--
## Evaluation after model traning (after running train.py)
Test:  [ 0/51]  eta: 0:00:02  model_time: 0.0299 (0.0299)  evaluator_time: 0.0030 (0.0030)  time: 0.0449  data: 0.0100  max mem: 465  
Test:  [50/51]  eta: 0:00:00  model_time: 0.0332 (0.0352)  evaluator_time: 0.0010 (0.0016)  time: 0.0482  data: 0.0104  max mem: 465  
Test: Total time: 0:00:02 (0.0484 s / it)  
Averaged stats: model_time: 0.0332 (0.0352)  evaluator_time: 0.0010 (0.0016)  
Accumulating evaluation results...  
DONE (t=0.02s).  
IoU metric: bbox  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.820  
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.981  
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.922  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.824  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.791  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.854  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.854  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.775  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.857  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.857  

## Evaluation after running test.py

Test:  [ 0/51]  eta: 0:04:47  model_time: 5.6357 (5.6357)  evaluator_time: 0.0020 (0.0020)  time: 5.6456  data: 0.0070  max mem: 181  
Test:  [50/51]  eta: 0:00:00  model_time: 0.0277 (0.1409)  evaluator_time: 0.0010 (0.0014)  time: 0.0405  data: 0.0065  max mem: 181  
Test: Total time: 0:00:07 (0.1497 s / it)  
Averaged stats: model_time: 0.0277 (0.1409)  evaluator_time: 0.0010 (0.0014)  
Accumulating evaluation results...  
DONE (t=0.06s).  
IoU metric: bbox  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.821  
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.981  
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.952  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.713  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.826  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.796  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.861  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.861  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.738  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.864  
