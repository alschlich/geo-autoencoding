import matplotlib.pyplot as plt
import os
import shutil

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)