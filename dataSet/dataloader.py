from torch.utils.data.dataset import Dataset
import torch
from skimage import io
from torchvision.transforms import Resize, Normalize

class GameDataSet(Dataset):
    def __init__(self, label_path):
        super(GameDataSet, self).__init__()
        self.Preprocessing = Resize([832, 480])
        self.Preprocessing2 = Normalize(mean=[128, 128, 128], std=[128, 128, 128])
        self.data = []
        with open(label_path, 'r') as label_file:
            for line in label_file:
                # print(line)

                words = line.split()
                self.data.append((words[0], [int(words[i]) for i in range(1, len(words))]))

    def __getitem__(self, index):
        fileName, label = self.data[index]
        img = io.imread(fileName)
        img = torch.from_numpy(img)
        # print(img.shape)
        with torch.no_grad():
            img = img.permute(2, 1, 0)
            img = self.Preprocessing(img)
            img = img.float()
            img = self.Preprocessing2(img)
        # label = torch.Tensor(label)
        # print(img.shape)
        return img, label

    def __len__(self):
        return len(self.data)


# if __name__ == "__main__":
#     train_data = GameDataSet('../data/train.txt')
#     for i in range(len(train_data)):
#         img, label = train_data[i]
#         print(img.shape)
#         print(label)
#         break
