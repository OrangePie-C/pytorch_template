import pdb
import os
import numpy as np
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn

import models
import tools.utils as utils
import tools.metrics as metrics
import tools.logging as logging

from tools.logging import AverageMeter, progress_bar

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions


def build_model(model_name, **kwargs):
    model_name = model_name.capitalize()
    model_type = getattr(models, model_name)
    return model_type(**kwargs)

# paramter setting
opt = TestOptions()
args = opt.initialize().parse_args()
print(args)
assert args.is_depth or args.is_seg

if args.do_save_result:
    save_path = os.path.join(args.result_dir, args.exp_name)
    logging.check_and_make_dirs(save_path)
    print("Saving result images in to %s" % save_path)

# device setting
if args.gpu_or_cpu == 'gpu':
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

print("====== 1. Defining model ===== ")

# model setting
model = build_model(args.model, is_seg=args.is_seg, is_depth=args.is_depth,
                    max_depth=args.max_depth, num_classes=args.num_classes,
                    dec_type=args.dec_type, backbone=args.backbone)

model.to(device)
model_weight = torch.load(args.ckpt_dir)

if 'module' in next(iter(model_weight.items()))[0]:
    model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())

model.load_state_dict(model_weight)
model.eval()

print("====== 2. Defining dataloader ===== ")
# dataloader setting
if args.dataset == 'ImagePath':
    kwargs = {'data_path': args.data_path}
else:
    kwargs = {'data_path': args.data_path, 'split': args.split, 'is_train': False}

test_dataset = get_dataset(args.dataset, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                          num_workers=args.workers, pin_memory=True)

print("===== 3. evaluate =====")
fps, pred = inference()

if args.do_evaluate:
    if args.is_depth:
        depth_metrics = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10',
                         'silog']
        for metric in depth_metrics:
            result_metrics['depth'] = {}
            result_metrics['depth'][metric] = 0.0

    if args.is_seg:
        seg_metrics = ['pixel_acc', 'mean_acc', 'mean_IoU']
        for metric in seg_metrics:
            result_metrics['seg'] = {}
            result_metrics['seg'][metric] = 0.0



def inference():
    elapsed_time = 0.0

    for batch_idx, (input_RGB, seg_gt, depth_gt, filename) in enumerate(test_loader):
        input_RGB = input_RGB.to(device)

        if args.gpu_or_cpu == 'gpu':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        with torch.no_grad():
            pred = model(input_RGB)

        if args.gpu_or_cpu == 'gpu':
            torch.cuda.synchronize()
            end.record()

        if args.gpu_or_cpu == 'gpu':
            elapsed_time += start.elapsed_time(end)

    fps = 1.0 / (elapsed_time / (batch_idx + 1) / 1000)

    return fps, pred

    #
    #print("Calculated fps: ", fps)



def test():
    elapsed_time = 0.0

    for batch_idx, (input_RGB, seg_gt, depth_gt, filename) in enumerate(test_loader):
        input_RGB = input_RGB.to(device)

        seg_gt, depth_gt = seg_gt.to(device), depth_gt.to(device)  # B x H x W

        if args.gpu_or_cpu == 'gpu':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        with torch.no_grad():
            pred = model(input_RGB)
            #pred_cpu = pred['pred_s'].cpu()
            #pred_cpu = pred['pred_d'].cpu()
        if args.gpu_or_cpu == 'gpu':
            torch.cuda.synchronize()
            end.record()

        if args.is_seg:
            pred_s = pred['pred_s']

        if args.is_depth:
            pred_d = pred['pred_d']

        if args.do_evaluate:
            if is_depth:
                pred_d = torch.reshape(pred_d, depth_gt.shape)
                computed_result = metrics.eval_depth(pred_d, depth_gt)
                for key in result_metrics['depth'].keys():
                    result_metrics['depth'][key] += computed_result[key]

        if args.do_save_result:
            pred_d = pred_d * 3
            logging.save_images(pred_d, save_result_path, filename[0], task_type='depth')
            logging.save_images(pred_s, save_result_path, filename[0], dataset='cityscapes', task_type='seg')

        progress_bar(batch_idx, len(test_loader), 1, 1)
        if args.gpu_or_cpu == 'gpu':
            elapsed_time += start.elapsed_time(end)

    for key in result_metrics['depth'].keys():
        result_metrics['depth'][key] = result_metrics['depth'][key] / (batch_idx + 1)

    fps = 1.0 / (elapsed_time / (batch_idx + 1) / 1000)
    print("Calculated fps: ", fps)
    display_result = logging.display_result(result_metrics)
    print(display_result)


if __name__ == "__main__":
    main()