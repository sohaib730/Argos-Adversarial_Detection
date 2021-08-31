
in_path = 'Raw_data/ImageNet/ILSVRC2012_img_train'
in_info_path = 'Raw_data/ImageNet/imagenet_info'

save_folder ='data/imagenet/'

from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import torchvision
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle


def Union(lst1, lst2):
    return list(set(lst1) | set(lst2))

in_hier = ImageNetHierarchy(in_path, in_info_path)
superclass_wnid = common_superclass_wnid('geirhos_16')
#superclass_wnid2 = common_superclass_wnid('mixed_13')
#superclass_wnid = Union(superclass_wnid1,superclass_wnid2)
#superclass_wnid.remove('n03405725')
#superclass_wnid.in
print (superclass_wnid)
class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)
print (label_map)

custom_dataset = datasets.CustomImageNet(str(in_path), class_ranges)
train_loader, test_loader = custom_dataset.make_loaders(workers=1,batch_size=1000,shuffle_train=True)

print(f"Train set size: {len(train_loader.dataset.targets)}")
print(f"Test set size: {len(test_loader.dataset.targets)}")



t_images = []
t_labels = []

images = []
labels = []
trans = torchvision.transforms.ToPILImage(mode = 'RGB')
Torch_Resize = torchvision.transforms.Resize((32,32), PIL.Image.NEAREST)
for d in train_loader:
    im,label = d
    im = torch.clamp(im,0,1.0)
    for k in range(im.shape[0]):
        r_im = Torch_Resize(trans(im[k]))
        images.append(np.array(r_im))
        labels.append(int(label[k])))

for d in test_loader:
    im,label = d
    im = torch.clamp(im,0,1.0)
    for k in range(im.shape[0]):
        r_im = Torch_Resize(trans(im[k]))
        t_images.append(np.array(r_im))
        t_labels.append(int(label[k])))



im_array = np.array(images)
lb_array = np.array(labels)
data = {'x':im_array,'y':lb_array}
pickle.dump(data,open(f"{save_folder}batch_1","wb"))

im_array = np.array(t_images)
lb_array = np.array(t_labels)
data = {'x':im_array,'y':lb_array}
pickle.dump(data,open(f"{save_folder}test","wb"))
