import os
import numpy as np
import matplotlib.pyplot as plt


def save_plots(fig_names, outputs, save_root):
    os.makedirs(save_root, exist_ok=True)
    h, w = outputs.shape[-2:]
    outputs = outputs.reshape(-1, 2, h, w)

    for (fig_name, output) in zip(fig_names, outputs):
        x, y = output
        plt.imshow(x, cmap='coolwarm', vmin=-1, vmax=1)
        plt.savefig(os.path.join(save_root, 'x_' + fig_name))
        plt.close()
        plt.imshow(y, cmap='coolwarm', vmin=-1, vmax=1)
        plt.savefig(os.path.join(save_root, 'y_' + fig_name))
        plt.close()
        
