from pickletools import TAKEN_FROM_ARGUMENT1
import pandas as pd
import numpy as np

def dict_Gordon(csv):
    df = pd.read_csv(csv) #read csv file

    length = len(df) # length 

    target_c = {}

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    boxes = []
    image_id = []
    class_id = []

    area = []

    for i in range(length):
        xmin.append(df.iat[i,1]) #xmin
        xmax.append(df.iat[i,2]) #xmax
        ymin.append(df.iat[i,3]) #ymin
        ymax.append(df.iat[i,4]) #ymax
        boxes.append([df.iat[i,1],df.iat[i,3],df.iat[i,2],df.iat[i,4]]) #xmin ymin xmax ymax
        image_id.append(df.iat[i,0]) #collection of '____.jpg'
        class_id.append(df.iat[i,5]) #TAKE NOTE OF [] SHEESH
        area.append((df.iat[i,2]-df.iat[i,1])*(df.iat[i,4]-df.iat[i,3]))

    #print(area)

    #u = np.unique(image_id) #unique count 
    #u[:] = u[::-1] #reverse

    a = np.array(image_id)
    _, idx = np.unique(a, return_index=True)
    #print(a[np.sort(idx)])
    store = a[np.sort(idx)]
    #print(store)

    #indexes = np.unique(image_id, return_index=True)[1]
    #store = [image_id[index] for index in sorted(indexes)]

    #print(type(store[1]))
    target_c = {} #big diCT lmao

    temp = [] #temporary storage for boxes indexing
    temp_labels = [] #temporary storage for labels indexing
    temp_imageid = [] #temporary storage for imageid indexing
    temp_area = [] #temporary storage for area indexing
    temp_iscrowd = [] #temporary storage for iscrowd indexing

    for i in range (len(store)): #populate temporary indexing storage withe empty lists
        temp.append([]) 
        temp_labels.append([])
        temp_imageid.append([])
        temp_area.append([])
        temp_iscrowd.append([])

    u_int = []

    for i in range(len(store)):
        ui = int(store[i].replace('.jpg',''))
        u_int.append(ui)

    #print("u_int",u_int)


    #print(store)
    #print("image_id",image_id)
    #print("u",u)
    #print("ui",ui)
    #print("u_int",u_int)


    #print(type(u_int[1]))
    #print("u_int:",u_int)
    #print(temp)
    #img_id
    #print(image_id)

    temp[0].append(boxes[0]) #initialize first element of boxes
    temp_labels[0].append(class_id[0]) #initialize second element of class id 
    temp_area[0].append(area[0])
    temp_iscrowd[0].append(0)

    #print(temp_iscrowd)

    j = 1
    for i in range(1,len(image_id)):
        if image_id[i] == image_id[i-1]:
            temp[j-1].append(boxes[i])
            temp_labels[j-1].append(class_id[i])
            temp_area[j-1].append(area[i])
            temp_iscrowd[j-1].append(0)
            j=j
        else:
            temp[j].append(boxes[i])
            temp_labels[j].append(class_id[i])
            temp_area[j].append(area[i])
            temp_iscrowd[j].append(0)
            j+=1

    #print(temp_iscrowd)

    #print(temp[1])

    for i in range (len(store)):
        #print(i) prints 0 to 995 
        target_c[store[i]] = {}
        target_c[store[i]]["boxes"] = temp[i]
        target_c[store[i]]["labels"] = temp_labels[i]
        target_c[store[i]]["image_id"] = u_int[i]
        target_c[store[i]]["area"] = temp_area[i]
        target_c[store[i]]["iscrowd"] = temp_iscrowd[i]
    
    return target_c

#dict_Gordon('drinks/labels_test.csv')