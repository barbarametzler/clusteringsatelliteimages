##open pickle and map to images in dataset

import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import PIL

data_set = '~/Imagery/WV2/multi/AMA/ReProj/Clipped_AMA_box/PCA/128/*/'
#data_set = '/home/bmetzler/Documents/Imagery/Accra/preprocessed/deepcluster/rgb_byte_scaled/'
#data_set = '/home/bmetzler/Documents/Imagery/Accra/preprocessed/kmeans/kmeans_4_tiles/'

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def all_tif_files(dirName):
    file_list = []
    # name of folder with imagery data
    #dirName = '/path/to/data_set';

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)

    # Print the files
    #for elem in listOfFiles:
    #    print(elem)

    #print ("****************")

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # Print the files
    for elem in listOfFiles:
        if elem.endswith(".tif"): #or .tif/png
            file_list.append(elem)
    return(file_list)


with open("/home/bmetzler/Documents/GitHub/deepcluster-master/exp/rgb_2000_vgg/clusters", "rb") as f:
    b = pickle.load(f)
    print (b[-1], max(b[-1]))


df = pd.DataFrame(all_tif_files(data_set), columns=['files'])
#print (df.head())

cluster_dict = dict(enumerate(b[-1]))
print (cluster_dict)
## plot 10 images of a certain cluster

###chose cluster to visualise:
cluster_number = 50
rows = 8
limit = 40
f, axarr = plt.subplots(3,3, figsize=(11,20))
f.tight_layout()


picture_list = cluster_dict[cluster_number]
print (len(picture_list))

for num, x in enumerate(picture_list):
    #print (str(df.loc[x][0]))
    if num>= limit: break
    img = PIL.Image.open(str(df.loc[x][0]))
    plt.subplot(rows,5,num+1)
    #plt.title(x.split('.')[0])
    plt.axis('off')
    plt.imshow(img)

plt.show()
