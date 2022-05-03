import os
import zipfile
import requests

#print("pip install pycocotools")

def download(url,path,filename, chunk_size=1024):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as downloaded:
        for chunk in r.iter_content(chunk_size=chunk_size):
            downloaded.write(chunk)

def extract():
    with zipfile.ZipFile("drinks.zip","r") as zip_ref:
        zip_ref.extractall()

def zipper():

    if os.path.exists('drinks.zip'):
        print("The zipped dataset file already exists in the current directory.")
        print("")
    else: 
        print("The zipped dataset file containing the trained model does not exist in the current directory.")
        print("Please wait while the zipped dataset file is being downloaded from the github release.")
        print("Downloading...")
        download("https://github.com/shawarmabytes/CoE-197Z-Deep-Learning-Object-Detection/releases/download/v1.0/drinks.zip", "drinks.zip", 'drinks.zip')
        print("Download finished.")
        print("")

    if os.path.exists('./drinks'):
        print("Dataset file 'drinks' is already extracted.")
        print("")
    else:
        print("Dataset 'drinks' folder does not exist in the current directory")
        print("Please wait while the zipped datset file is being extracted.")
        print("Preparing to unzip file...")
        print("Unzipping...")
        extract()
        print("Extract finished.")
        print("")

def pth_get():
    if os.path.exists('drinks_dataset_trained_model.pth'):
        print("The file containing the trained model already exists in the current directory.")
        print("Initializing testing ...")
    else:
        print("The file containing the trained model does not exist in the current directory.")
        print("Please wait while the file is being downloaded from the github release.")
        print("Downloading...")
        download("https://github.com/shawarmabytes/CoE-197Z-Deep-Learning-Object-Detection/releases/download/v1.0/drinks_dataset_trained_model.pth", 'drinks_dataset_trained_model.pth',"drinks_dataset_trained_model.pth")
        print("Download finished. Initializing testing...")



'''
if os.path.exists('drinks_dataset_trained_model.pth'):
    print("The file containing the trained model is already downloaded. Initializing testing ...")
else:
    print("The file containing the trained model is not yet downloaded.")
    print("Downloading ...")
    download("https://github.com/shawarmabytes/CoE-197Z-Deep-Learning-Object-Detection/releases/download/v1.0/drinks_dataset_trained_model.pth", 'drinks_dataset_trained_model.pth',"drinks_dataset_trained_model.pth")
    print("Download finished. Initializing testing...")
'''
'''
def get_file(url,path,filename, chunk_size=1024):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as downloaded:
        for chunk in r.iter_content(chunk_size=chunk_size):
            downloaded.write(chunk)

get_file("https://github.com/shawarmabytes/CoE-197Z-Deep-Learning-Object-Detection/releases/download/v1.0/drinks_dataset_trained_model.pth", 'drinks_dataset_trained_model.pth',"drinks_dataset_trained_model.pth")
'''


'''
print(os.path)
print(os.getcwd()+'/drinks')
x = (os.getcwd()+'/drinks')
print(os.path.exists('train.py'))
'''