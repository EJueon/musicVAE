from typing import List

import os
import pandas as pd

def load_files(dir_path: str, extensions: List[str] = []):
    """Load files with specific extensions from directory

    Args:
        dir_path (str): Path of root directory
        extensions (List(str), optional): List of file extensions. Defaults to [].
    """

    for dir_path, _, file_names in os.walk(dir_path):
        for file_name in file_names:
            if file_name.endswith(tuple(extensions)):
                yield os.path.abspath(os.path.join(dir_path, file_name))


def save_dataset(dataset:List, save_path:str='./', dev_ratio:float = 0.2, header:List=["filepath", "data"]):
    df = pd.DataFrame(dataset, columns = header)
    dev_dataset = df.sample(n=int(len(df) * dev_ratio)).reset_index(drop=True)
    train_dataset = df.drop(dev_dataset.index).reset_index(drop=True)
    
    train_dataset_path = os.path.join(save_path, "train_data.json")
    dev_dataset_path = os.path.join(save_path, "dev_data.json")
    train_dataset.to_json(train_dataset_path, orient="records", force_ascii=False, lines=True)
    dev_dataset.to_json(dev_dataset_path, orient="records", force_ascii=False, lines=True)
    return train_dataset_path, dev_dataset_path

def load_dataset(dataset_path: str) -> pd.DataFrame:
    return pd.read_json(dataset_path, lines=True)

def setdir(dirpath: str, dirname: str = None, reset: bool = False):
    from shutil import rmtree

    filepath = os.path.join(dirpath, dirname) if dirname else dirpath
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    elif reset:
        rmtree(filepath)
        os.mkdir(filepath)
    return filepath