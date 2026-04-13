# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/17 20:15
Create User : 19410
Desc : xxx
"""
import numpy as np
from torch.utils.data import Dataset, DataLoader

np.random.seed(28)


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.x = np.random.randn(100, 2)
        self.y = np.random.randint(0, 2, size=(100,))

    def __getitem__(self, item):
        # return self.x[item], self.y[item]
        return {
            "x": self.x[item],
            "y": self.y[item]
        }

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    ds = MyDataset()
    print(f"数据集大小:{len(ds)}")
    print(f"第五条数据为:{ds[5]}")

    dataloader = DataLoader(
        dataset=ds,  # 数据集
        batch_size=4,  # 将几个样本合并成一个批次
        shuffle=True,  # 加载数据的时候是否打乱顺序
        # num_workers=0,  # 给定使用多少个进程进行数据加载
        collate_fn=None,  # 给定从dataset获取的多个数据如何合并成一个批次数据 入参就是list[item] item实际上就是dataset返回的数据对象
    )
    for batch in dataloader:
        print(batch)
        break
