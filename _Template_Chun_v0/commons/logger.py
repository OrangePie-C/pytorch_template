import itertools
import logging
import os

import colored
import numpy as np
from PIL import Image
from colored.colored import stylize


formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 0
        logger.handlers.clear()
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    print('logger(%s) handler #=%d' % (name, len(logger.handlers)))
    logger.setLevel(logging.DEBUG)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)




def print_log(predictions, corrects=None):
    max_val = max(predictions)
    colors = [237, 237, 237, 237, 240, 243, 246, 249, 252, 255]
    log = ''
    for idx, p in enumerate(predictions):
        color_id = int((p / max_val) * 10 + 0.5)
        color_id = min(9, color_id)
        color_id = max(0, color_id)
        if corrects is not None:
            c = corrects[idx]
        else:
            c = ''
        log += stylize('%.3f%s ' % (p, c), colored.fg(colors[color_id]))
    argmax_id = np.argmax(predictions.detach().cpu().numpy())
    log += str(decode_desc(argmax_id))
    print(log)


import torch
from tensorboardX.utils import make_grid
from dataset.cityscapes_labels import trainId2color

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, epochs, cur_epoch, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    remain_time = step_time * (total - current) + (epochs - cur_epoch) * step_time * total

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    L.append(' | Rem: %s' % format_time(remain_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


class AverageMeter():
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes).zfill(2) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf).zfill(2) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis).zfill(3) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def display_result(result_dict):
    line = "\n"
    line += "=" * 100 + '\n'
    for task, task_result in result_dict.items():
        line += "%s\n" % task
        for metric in task_result.keys():
            line += "{:>10} ".format(metric)
        line += "\n"
        for metric_result in task_result.values():
            line += "{:10.4f} ".format(metric_result)
        line += "\n"
    line += "=" * 100 + '\n'

    return line

def display_best_result(result_dict):
    line = ""
    for task ,task_result in result_dict.items():
        line += "Best %s results: epoch %d\n" % (task, task_result['epoch'])
        for metric in task_result['best'].keys():
            line += "{:>10} ".format(metric)
        line += "\n"
        for metric_result in task_result['best'].values():
            line += "{:10.4f} ".format(metric_result)
        line += "\n"
    line += "=" * 100 + '\n'

    return line

def seg_label_to_color(pred, dataset):
    '''
    :param pred: [B x Class x H x W]
    :param dataset: 'cityscapes'
    :return: [B x 3 x H x W]
    '''
    if dataset == 'cityscapes':
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 3)).astype(np.uint8)
        for k, v in trainId2color.items():
            pred_color[pred == k, :] = v
        cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
    else:
        pred_color = cv2.applyColorMap(pred, cv2.COLORMAP_RAINBOW)

    return pred_color

def save_images(pred, save_path):
    if len(pred.shape) > 3:
        pred = pred.squeeze()

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy().astype(np.uint8)

    if pred.shape[0] < 4:
        pred = np.transpose(pred, (1, 2, 0))
    cv2.imwrite(save_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def check_and_make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
