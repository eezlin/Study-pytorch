# csv(id, feat1, feat2, label)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class ExcelDataset(Dataset):
    def __init__(self, excel_file="data.xlsx", sheet_name=0):
        df = pd.read_excel(
            excel_file, header=0, index_col=0,
            names=["feat1", "feat2", "label"],
            sheet_name=sheet_name,
            dtype={"feat1": np.float32, "feat2": np.float32, "label": np.int32})

        feat = df.iloc[:, :2].values
        label = df.iloc[:, 2].values

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class CsvDataset(Dataset):
    def __init__(self, csv_file="data.csv"):
        df = pd.read_csv(
            csv_file, header=0, index_col=0,
            encoding="utf-8",
            names=["feat1", "feat2", "label"],
            dtype={"feat1": np.float32, "feat2": np.float32, "label": np.int32},
            skip_blank_lines=True)

        feat = df.iloc[:, :2].values
        label = df.iloc[:, 2].values

        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]