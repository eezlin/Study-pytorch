from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.path = os.path.join(self.data, self.labels)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.data, self.labels, img_name)
        img = Image.open(img_item_path)
        label = self.labels
        return img, label
    def __len__(self):
        return len(self.img_path)

data = "dataset/train"
ants_labels = "ants"
bees_labels = "bees"
ants_dataset = MyDataset(data, ants_labels)
bees_dataset = MyDataset(data, bees_labels)
train_dataset = ants_dataset + bees_dataset