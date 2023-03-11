import numpy as np
import PIL
import io
import base64
from IPython.display import Image as Image, clear_output
import matplotlib.pyplot as plt


def plot_train_log(log_dict, plt_number, save_path):
    plt.figure(figsize=(6 * plt_number, 6))
    plt_idx = 1
    for log_name in log_dict.keys():
        loss_log = log_dict[log_name][0]
        if(len(loss_log) == 0):
            continue
        y_scale = log_dict[log_name][1]
        ylim = log_dict[log_name][2]

        plt.subplot(1, plt_number, plt_idx)
        plt.plot(loss_log, '.', alpha=0.1)
        plt.title(log_name)
        if (y_scale):
            plt.yscale('log')
        if (ylim):
            plt.ylim(np.min(loss_log), max(loss_log))
        plt_idx += 1
    plt.savefig(save_path)
    


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)


def imwrite(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
        f = open(f, 'wb')
    np2pil(a).save(f, fmt, quality=95)


def imencode(a, fmt='jpeg'):
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = 'png'
    f = io.BytesIO()
    imwrite(f, a, fmt)
    return f.getvalue()


def im2url(a, fmt='jpeg'):
    encoded = imencode(a, fmt)
    base64_byte_string = base64.b64encode(encoded).decode('ascii')
    return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string


def imshow(a, fmt='jpeg', _clear_output=False):
    if _clear_output:
        clear_output(True)
    display(Image(data=imencode(a, fmt)))


def save_train_image(imgs, save_path, return_img = False):
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = np.hstack(imgs)
    imgs = np2pil(imgs)
    if(return_img):
        return imgs
    imgs.save(save_path, quality = 100)
    
