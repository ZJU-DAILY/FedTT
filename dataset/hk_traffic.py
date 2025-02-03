from dataset.dataset import Dataset
import numpy as np
import pandas as pd


class PeMSD4(Dataset):
    def __init__(self, args):
        super().__init__(args=args)
        self.data = np.load(f'hk.npy')['data'].astype('float32')
        self.road_network = pd.read_csv("distances.csv")
