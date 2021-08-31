
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import csv
import numpy as np
import pickle

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels
mean = np.zeros((32,32,3))

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    global mean
    for c in range(0,43):
        print ("Loading Label",c)
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            im = Image.open(prefix + row[0]) # the 1th column is the filename

            if im.mode != "RGB":
                im = im.convert(mode="RGB")
            im = ImageOps.equalize(im, mask = None)
            crop_rectangle = (int(row[3])-1, int(row[4])-1, int(row[5])+1, int(row[6])+1)
            cropped_im = im.crop(crop_rectangle)
            im_resized = cropped_im.resize((32, 32), Image.BOX)
            im_resized = np.array(im_resized)
            #mean = mean +  im_resized
            images.append(im_resized)
            labels.append(int(row[7])) # the 8th column is the label
        gtFile.close()
        print (len(labels))
    #mean = mean / len(labels)
    #print (mean)
    return images , labels

def read_Test_TrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels

    #prefix = rootpath + '/Images/'
    gtFile = open(rootpath + 'GT-final_test'+ '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    # loop over all images in current annotations file

    for row in gtReader:
        im = Image.open(prefix + row[0]) # the 1th column is the filename

        if im.mode != "RGB":
            im = im.convert(mode="RGB")

        im = ImageOps.equalize(im, mask = None)
        crop_rectangle = (int(row[3])-1, int(row[4])-1, int(row[5])+1, int(row[6])+1)
        cropped_im = im.crop(crop_rectangle)
        im_resized = cropped_im.resize((32, 32), Image.BOX)
        images.append(np.array(im_resized))
        labels.append(int(row[7])) # the 8th column is the label
    gtFile.close()
    return images, labels

def display(image,label):
    width=3
    height=3
    rows = 10
    cols = 10
    axes=[]
    fig=plt.figure(figsize=(15,15))
    #plt.figure(f)
    for a in range(cols):
        ind = np.where(label == a)
        #print (ind[0])
        b = image[ind[0]]
        for r in range(rows):
            axes.append( fig.add_subplot(rows, cols, a+(r*cols)+1 ))
            subplot_title=(a)

            axes[-1].set_title(subplot_title)
            plt.imshow(b[r])
            plt.axis("off")
    fig.tight_layout()
    plt.savefig("data.png")
if __name__ == "__main__":
    #print (mean)
    ###Reading Train Data
    train_folder_path = 'Raw_data/GTSRB/Final_Training/Images'
    test_folder_path = 'Raw_data/GTSRB/Final_Test/Images'

    save_folder = 'data/gtsrb/'
    im,lab = readTrafficSigns(train_folder_path)

    im = np.array(im,dtype = 'uint8')

    lab = np.array(lab)

    perm = np.arange(im.shape[0])
    np.random.shuffle(perm)
    image = im[perm]
    label = lab[perm]

    #image = np.squeeze(im)
    #lab = np.squeeze(lab)
    print(label.shape)
    print (image.shape)
    print (label[0:10])

    #display(image,label)

    ###Storing Train/Validation data in Batch format
    for i in range(5):
        start = i*7000
        end = (i+1) * 7000
        d = {'data':image[start:end],'label':label[start:end]}
        pickle.dump(d, open(f"{save_folder}batch_{i+1}", "wb"))

    #d = {'data':image[35000:],'labels':label[35000:]}
    #pickle.dump(d, open(f"{save_folder}val", "wb"))



    ###Readin Test data
    im,lab = read_Test_TrafficSigns(test_folder_path)
    im = np.array(im)
    lab = np.array(lab)

    #image = np.squeeze(im)
    #lab = np.squeeze(lab)
    print(lab.shape)
    print (im.shape)
    print ("Test Label",lab[0:10])
    d = {'data':im,'label':lab}
    pickle.dump(d, open(f"{save_folder}test", "wb"))
