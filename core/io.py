import io
import pickle

import numpy as np
import torch


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
          return super().find_class(module, name)


def save_pickle(filepath, content):
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(content, pickle_file, -1)


def load_pickle(filepath):
    with open(filepath, "rb") as input_file:
        annotations = CPU_Unpickler(input_file).load()
    return annotations


def load_annotation(annotation_filename):
    """
    Load annotations file to use for labeling
    """
    annotations = load_pickle(annotation_filename)
    return annotations


def load_depth(filepath):
    """
    Loads depth map and handles NaN/Inf values.
    """
    depth = np.load(filepath)
    depth = np.nan_to_num(depth, posinf=0.0, neginf=0.0, nan=0.0)
    return depth