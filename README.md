# CoE-197Z-Deep-Learning-Keyword-Spotting

## Author
Josh Lear Yap  
2018-0XXX7  
BS Electronics Engineering  
University of the Philippines Diliman  
Electrical and Electronics Engineering Institute  

## References
[Keyword Spotting (KWS) using PyTorch Lightning](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws_demo.ipynb)  
[Keyword Spotting (KWS) application](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws-infer.py)  
[Transformer for CIFAR10](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/transformer/python/transformer_demo.ipynb)  

## About the project
This project used 

## Performance Metrics
The training accuracy reached 92.80603% with 16 patches. For more information about this, visit the [wandb logs here,](https://wandb.ai/shawarmabytes/pl-kws/runs/1dd0x972/logs?workspace=user-shawarmabytes) and the [run overview here.](https://wandb.ai/shawarmabytes/pl-kws/runs/1dd0x972/overview?workspace=user-shawarmabytes)

## Python Scripts

``trainer.py`` trains the speech-command dataset using a transformer model  

``transformer_model`` contains the transformer model suitable for the speech-commands dataset

``dataset.py`` contains the LightningDataModule suitable for the transformer model  

``kws-infer.py`` prompts a microphone based GUI for keyword spotting application  

## Prerequisite/Setup
### Install requirements
```
pip3 install -r requirements.txt
```
Note: The requirements.txt file assumes that you have torch and torchvision installed with cuda enabled.

Otherwise, you can install them by running:

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## Code execution

### Training the transformer model 
```
python trainer.py
```

### Keyword Spotting Application
```
python kws-infer.py
```

## Machine GPU used
NVIDIA GeForce RTX 2060



