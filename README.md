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

## Python Scripts

``trainer.py`` trains the speech-command dataset using a transformer model  

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


