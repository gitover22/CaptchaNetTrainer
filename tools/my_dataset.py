import os

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tools import captcha_info
from PIL import Image
from tools import trans

transform = transforms.Compose([
    transforms.Resize((80, 200)),
    transforms.Grayscale(), # dimensionality
    transforms.ToTensor()
])


class mydataset(Dataset):
    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        # print(image_name)
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = trans.encode(image_name.split('_')[0])
        return image, label


def Get_train_Dataloader():
    dataset = mydataset(captcha_info.Train_Dataset, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)


def Get_test_Dataloader():
    dataset = mydataset(captcha_info.Test_Dataset, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


if __name__ == '__main__':
    '''
    test
    '''
