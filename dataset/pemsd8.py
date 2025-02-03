from dataset.dataset import Dataset
import numpy as np
import pandas as pd


class PeMSD8(Dataset):
    def __init__(self, args):
        super().__init__(args=args)
        self.data = np.load(f'pemsd8.npz')['data'].astype('float32')
        self.road_network = pd.read_csv("pemsd8.csv")
