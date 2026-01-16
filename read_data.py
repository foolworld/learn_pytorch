from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):

    def __init__(self,root_dir,laber_dir):
        self.root_dir = root_dir
        self.laber_dir = laber_dir
        self.path = os.path.join(self.root_dir,self.laber_dir)
        self.image_path = os.path.(self.path)
        self.label_path = os.path.join(self.root_dir,self.laber_dir)



    def __getitem__(self, index):
        img_name = self.image_path[index]
        img_item_path = os.path.join(self.root_dir,img_name)
        img = Image.open(img_item_path)

    def __len__(self):
        return len(self.image_path)

