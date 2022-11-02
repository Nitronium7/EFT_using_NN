from torch.utils.data import Dataset
import numpy as np
import torch

class dataset_from_df(Dataset):
    def __init__(self, arr, GPU):
        enable_gpu = GPU
        if enable_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        x = arr[:, :-1]
        y = arr[:, -1]
        y = y.astype(np.int64)

        self.x_train = torch.tensor(x, dtype=torch.float32).to(device)
        self.y_train = torch.tensor(y, dtype=torch.int64).to(device)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
