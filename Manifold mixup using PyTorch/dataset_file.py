import numpy as np 
from torch.utils.data import Dataset


class CharData(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.X = images
        self.y = labels
        
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = np.array(self.X.iloc[idx, :], dtype='uint8').reshape([28, 28, 1])
        if self.transform is not None:
            img = self.transform(img)
        
        if self.y is not None:
            y = np.zeros(10, dtype='float32')
            y[self.y.iloc[idx]] = 1
            return img, y
        else:
            return img
            