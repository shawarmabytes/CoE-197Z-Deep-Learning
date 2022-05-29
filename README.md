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

Epoch 8, global step 12618: 'test_acc' was not in top 1
Epoch 10:  90%|███████████████████████████████████████████████▊     | 1406/1558 [02:24<00:15,  9.72it/s, loss=0.428, v_num=x972, test_loss=0.450, test_acc=88.00]
Epoch 9, global step 14020: 'test_acc' was not in top 1
Epoch 11:  91%|████████████████████████████████████████████████▉     | 1411/1558 [02:25<00:15,  9.71it/s, loss=0.36, v_num=x972, test_loss=0.408, test_acc=89.80]
Epoch 10, global step 15422: 'test_acc' was not in top 1
Epoch 12:  91%|████████████████████████████████████████████████▊     | 1410/1558 [02:23<00:15,  9.82it/s, loss=0.42, v_num=x972, test_loss=0.371, test_acc=90.00]
Epoch 11, global step 16824: 'test_acc' was not in top 1
Epoch 13:  90%|███████████████████████████████████████████████▊     | 1404/1558 [02:23<00:15,  9.78it/s, loss=0.405, v_num=x972, test_loss=0.591, test_acc=83.10]
Epoch 12, global step 18226: 'test_acc' was not in top 1
Epoch 14:  91%|████████████████████████████████████████████████▎    | 1421/1558 [02:25<00:13,  9.79it/s, loss=0.281, v_num=x972, test_loss=0.443, test_acc=88.80]
Epoch 13, global step 19628: 'test_acc' was not in top 1
Epoch 15:  92%|████████████████████████████████████████████████▌    | 1426/1558 [02:28<00:13,  9.61it/s, loss=0.255, v_num=x972, test_loss=0.355, test_acc=90.80]
Epoch 14, global step 21030: 'test_acc' reached 90.77885 (best 90.77885), saving model to 'C:\\Users\\JOLEK\\Desktop\\data\\speech_commands\\checkpoints\\shawarmabytes-kws-best-acc.ckpt' as top 1
Epoch 16:  91%|███████████████████████████████████████████████▉     | 1411/1558 [02:33<00:15,  9.21it/s, loss=0.197, v_num=x972, test_loss=0.416, test_acc=89.70]
Epoch 15, global step 22432: 'test_acc' was not in top 1
Epoch 17:  92%|█████████████████████████████████████████████████▍    | 1427/1558 [02:34<00:14,  9.25it/s, loss=0.24, v_num=x972, test_loss=0.337, test_acc=92.00]
Epoch 16, global step 23834: 'test_acc' reached 91.96321 (best 91.96321), saving model to 'C:\\Users\\JOLEK\\Desktop\\data\\speech_commands\\checkpoints\\shawarmabytes-kws-best-acc.ckpt' as top 1
Epoch 18:  91%|████████████████████████████████████████████████▏    | 1415/1558 [02:32<00:15,  9.25it/s, loss=0.254, v_num=x972, test_loss=0.401, test_acc=90.10]
Epoch 17, global step 25236: 'test_acc' was not in top 1
Epoch 19:  90%|███████████████████████████████████████████████▊     | 1406/1558 [02:32<00:16,  9.22it/s, loss=0.244, v_num=x972, test_loss=0.473, test_acc=88.40]
Epoch 18, global step 26638: 'test_acc' was not in top 1
Epoch 20:  90%|███████████████████████████████████████████████▊     | 1404/1558 [02:31<00:16,  9.24it/s, loss=0.163, v_num=x972, test_loss=0.363, test_acc=91.10]
Epoch 19, global step 28040: 'test_acc' was not in top 1
Epoch 21:  91%|████████████████████████████████████████████████▎    | 1422/1558 [02:33<00:14,  9.25it/s, loss=0.198, v_num=x972, test_loss=0.343, test_acc=91.70]
Epoch 20, global step 29442: 'test_acc' was not in top 1
Epoch 22:  91%|████████████████████████████████████████████████▏    | 1416/1558 [02:32<00:15,  9.26it/s, loss=0.209, v_num=x972, test_loss=0.373, test_acc=91.50]
Epoch 21, global step 30844: 'test_acc' was not in top 1
Epoch 23:  90%|███████████████████████████████████████████████▊     | 1405/1558 [02:32<00:16,  9.21it/s, loss=0.153, v_num=x972, test_loss=0.342, test_acc=92.10]
Epoch 22, global step 32246: 'test_acc' reached 92.06337 (best 92.06337), saving model to 'C:\\Users\\JOLEK\\Desktop\\data\\speech_commands\\checkpoints\\shawarmabytes-kws-best-acc.ckpt' as top 1
Epoch 24:  91%|████████████████████████████████████████████████▎    | 1420/1558 [02:33<00:14,  9.26it/s, loss=0.191, v_num=x972, test_loss=0.321, test_acc=92.40]
Epoch 23, global step 33648: 'test_acc' reached 92.37436 (best 92.37436), saving model to 'C:\\Users\\JOLEK\\Desktop\\data\\speech_commands\\checkpoints\\shawarmabytes-kws-best-acc.ckpt' as top 1
Epoch 25:  91%|████████████████████████████████████████████████▎    | 1422/1558 [02:32<00:14,  9.30it/s, loss=0.163, v_num=x972, test_loss=0.332, test_acc=92.30]
Epoch 24, global step 35050: 'test_acc' was not in top 1
Epoch 26:  91%|████████████████████████████████████████████████▏    | 1416/1558 [02:33<00:15,  9.25it/s, loss=0.141, v_num=x972, test_loss=0.335, test_acc=92.70]
Epoch 25, global step 36452: 'test_acc' reached 92.69634 (best 92.69634), saving model to 'C:\\Users\\JOLEK\\Desktop\\data\\speech_commands\\checkpoints\\shawarmabytes-kws-best-acc.ckpt' as top 1
Epoch 27:  90%|███████████████████████████████████████████████▊     | 1406/1558 [02:32<00:16,  9.20it/s, loss=0.176, v_num=x972, test_loss=0.329, test_acc=92.60]
Epoch 26, global step 37854: 'test_acc' was not in top 1
Epoch 28:  91%|████████████████████████████████████████████████▍    | 1424/1558 [02:33<00:14,  9.25it/s, loss=0.138, v_num=x972, test_loss=0.327, test_acc=92.80]
Epoch 27, global step 39256: 'test_acc' reached 92.80603 (best 92.80603), saving model to 'C:\\Users\\JOLEK\\Desktop\\data\\speech_commands\\checkpoints\\shawarmabytes-kws-best-acc.ckpt' as top 1
Epoch 29:  91%|████████████████████████████████████████████████▍    | 1425/1558 [02:32<00:14,  9.35it/s, loss=0.116, v_num=x972, test_loss=0.335, test_acc=92.70]
Epoch 28, global step 40658: 'test_acc' was not in top 1
Testing DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 172/172 [00:14<00:00, 11.84it/s]
Epoch 29, global step 42060: 'test_acc' was not in top 1
Testing: 0it [00:00, ?it/s]
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
C:\Users\JOLEK\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:487: PossibleUserWarning: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test/predict dataloaders.
  rank_zero_warn(
C:\Users\JOLEK\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\pytorch_lightning\trainer\connectors\data_connector.py:240: PossibleUserWarning: The dataloader, test_dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        test_acc             91.93090057373047
        test_loss           0.3486728072166443
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
