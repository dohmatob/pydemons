import numpy as np
import matplotlib.pyplot as plt
from pydemons import demons


def run_demons(moving, fixed, **kwargs):
    # plot input images
    plt.ion()
    plt.figure(figsize=(13.5, 7))
    plt.gray()
    ax = plt.subplot(141)
    ax.set_title("fixed")
    plt.axis("off")
    ax.imshow(fixed)
    ax = plt.subplot(142)
    ax.set_title("moving")
    plt.axis("off")
    ax.imshow(moving)

    # run demons
    warped = moving
    diff = warped - fixed
    ax = plt.subplot(143)
    ax.set_title("warped")
    ax.axis("off")
    warped_thumb = ax.imshow(warped)
    ax = plt.subplot(144)
    ax.set_title("diff")
    ax.axis("off")
    diff_thumb = ax.imshow(diff)
    plt.show()

    def _callback(variables):
        warped = variables["warped"]
        fixed = variables['fixed']
        diff = warped - fixed
        warped_thumb.set_data(warped)
        plt.draw()
        diff_thumb.set_data(diff)
        plt.draw()

    return demons(fixed, moving, callback=_callback, **kwargs)

if __name__ == "__main__":
    # load data
    from PIL import Image
    fixed = np.array(Image.open("data/lenag2.png"), dtype=np.float)
    moving = np.array(Image.open("data/lenag1.png"),
                      dtype=np.float)
    run_demons(moving, fixed)
