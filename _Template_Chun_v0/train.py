import pdb
import os
import cv2
import time
import numpy as np
from datetime import datetime
#from model_chun import DDRNet_23_slim_Chun
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter
#from tensorboardX.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP

#import models
#import tools.utils as utils
import tools.logging as logging
import tools.metrics as metrics
import tools.criterion as criterion
from tools.logging import AverageMeter, progress_bar

from dataset.base_dataset import get_dataset

from torch.utils.data import DataLoader
# custom
from config import config, configuration
from dataset import data_loader
from torch.utils.tensorboard import SummaryWriter


# def build_model(model_name, **kwargs):
#     model_name = model_name.capitalize()
#     model_type = getattr(models, model_name)
#     return model_type(**kwargs)
#
#
# def build_criterion(criterion_name, **kwargs):
#     criterion_type = getattr(criterion, criterion_name + 'Loss')
#     return criterion_type(**kwargs)


def main():

    ## configuration / logger
    args = config.Load_Config.parse_args()
    logger = get_logger('learn2test-train')



    # initialize
    #writer = SummaryWriter()
    TBwriter = TensorboardWriter(args.log_dir, logger, args.TBboolean)

    # TBwriter.add_image('four_fashion_mnist_images', img_grid)
    # TBwriter.add_graph(net, images)
    # TBwriter.close()
    # writer.add_scalar('training loss',
    #                   running_loss / 1000,
    #                   epoch * len(trainloader) + i)
    #
    # writer.add_pr_curve(classes[class_index],
    #                     tensorboard_truth,
    #                     tensorboard_probs,
    #                     global_step=global_step)
    # writer.close()
    #
    # writer.add_figure('predictions vs. actuals',
    #                   plot_classes_preds(net, inputs, labels),
    #                   global_step=epoch * len(trainloader) + i)

    # dataloader
    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # network options



    # training options # criterion, metric, schdule, optimizer
    # criterion
    # sceduler
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.)
    # scheduler



    # log printer

    # train

    # val


    #######################################################################
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    assert args.is_depth or args.is_seg, \
        "you should choose at least one task between [--is_depth, --is_seg]"

    if args.is_depth and args.is_seg:
        # case sensitive
        assert 'Two' in args.dec_type, \
            "In case of you train joint learning, decoder name should include Two"

    if args.dataset == 'nyudepthv2':
        args.max_depth = 10.0
        args.num_classes, args.ignore_index = 40, 40
    print(args)

    # Logging
    exp_name = '%s%s%s_%s' % (datetime.now().strftime('%m%d'),
                              '_seg' if args.is_seg else '', '_depth' if args.is_depth else '', args.exp_name)
    log_dir = os.path.join(args.log_dir, args.dataset, args.backbone, exp_name)
    logging.check_and_make_dirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)

    log_txt = os.path.join(log_dir, 'logs.txt')
    if not os.path.exists(log_txt):
        with open(log_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'

            txtfile.write(args_str + '\n')


    if args.do_save_result:
        if args.is_seg:
            logging.check_and_make_dirs(os.path.join(log_dir, 'seg_outputs'))
        if args.is_depth:
            logging.check_and_make_dirs(os.path.join(log_dir, 'depth_outputs'))

    # Model and Dataloader
    # model = build_model(args.model, is_seg=args.is_seg, is_depth=args.is_depth, max_depth=args.max_depth,
    #                     num_classes=args.num_classes, dec_type=args.dec_type, backbone=args.backbone)
    model = DDRNet_23_slim_Chun.DualResNet(DDRNet_23_slim_Chun.BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64)
    num_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6  # to scale Million
    print('Number of model parameters: %.2fM' % num_params)

    # CPU-GPU agnostic settings
    if args.gpu_or_cpu == 'gpu':
        device = torch.device('cuda')
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    # args.split
    # val: train/val
    # test: trainval / test
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path':args.data_path, 'split':args.split}
    if args.dataset == 'cityscapes':
        dataset_kwargs['depth_type'] = args.depth_type
    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    # Training settings
    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion_dict = {}
    # Log best results
    best_dict = {}
    if args.is_seg:
        criterion_dict['criterion_s'] = build_criterion(args.criterion_s, ignore_index=args.ignore_index)
        best_dict['seg'] = {}
    if args.is_depth:
        criterion_dict['criterion_d'] = build_criterion(args.criterion_d)
        best_dict['depth'] = {}

    # Perform experiment
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        losses_dict = train(train_loader, model, criterion_dict, optimizer=optimizer, device=device,
                            epoch=epoch, args=args)
        for loss_name, loss_value in losses_dict.items():
            writer.add_scalar(loss_name, loss_value, epoch)

        if epoch % args.val_freq == 0:
            display_dict, results_dict, fps = validate(val_loader, model, num_classes=args.num_classes, device=device,
                                                       epoch=epoch, args=args, log_dir=log_dir)

            # best_dict['seg']['best']
            # results_dict['Segmentation results'] = {'pixel acc': , ...}

            # log best results
            if 'seg' in best_dict:
                key_metric = 'mean_IoU'
                if 'best' in best_dict['seg'].keys():
                    best_value = best_dict['seg']['best'][key_metric]
                    result_value = results_dict['Segmentation results'][key_metric]
                    if best_value < result_value:
                        best_dict['seg']['best'] =  results_dict['Segmentation results']
                        best_dict['seg']['epoch'] = epoch
                else:
                    best_dict['seg']['best'] = results_dict['Segmentation results']
                    best_dict['seg']['epoch'] = epoch

            if 'depth' in best_dict:
                key_metric = 'd1'
                if 'best' in best_dict['depth'].keys():
                    best_value = best_dict['depth']['best'][key_metric]
                    result_value = results_dict['Depth results'][key_metric]
                    if best_value < result_value:
                        best_dict['depth']['best'] = results_dict['Depth results']
                        best_dict['depth']['epoch'] = epoch
                else:
                    best_dict['depth']['best'] = results_dict['Depth results']
                    best_dict['depth']['epoch'] = epoch

            result_lines = logging.display_result(results_dict)
            best_lines= logging.display_best_result(best_dict)
            print(result_lines)
            print(best_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write('\nfps: %.5f' % fps)
                txtfile.write(result_lines)
                txtfile.write(best_lines)

            writer.add_scalar("fps", fps, epoch)

            if args.do_save_result:
                if args.is_seg:
                    save_path = os.path.join(log_dir, 'seg_outputs', '%03d_seg.png' % epoch)
                    out = make_grid(display_dict['pred_s'], ncols=2)
                    logging.save_images(out, save_path)

                if args.is_depth:
                    save_path = os.path.join(log_dir, 'depth_outputs', '%03d_depth.png' % epoch)
                    out = make_grid(display_dict['pred_d'], ncols=2)  # C x H x W
                    logging.save_images(out, save_path)

            for all_results in results_dict.values():
                for each_metric, each_results in all_results.items():
                    writer.add_scalar(each_metric, each_results, epoch)


def train(train_loader, model, criterion_dict, optimizer, device, epoch, args):
    model.train()
    losses_dict = {'seg_loss': AverageMeter(), 'depth_loss': AverageMeter()}
    for batch_idx, (input_RGB, seg_gt, depth_gt, filename) in enumerate(train_loader):
        seg_gt, depth_gt = seg_gt.to(device), depth_gt.to(device)  # B x H x W
        input_RGB = input_RGB.to(device)
        preds, preds_d = model(input_RGB)  # pred_s: [B x 1 x 512 x 1024]
        optimizer.zero_grad()

        if preds is not None:
            pred_s = preds.squeeze()
            loss_s = criterion_dict['criterion_s'](pred_s, seg_gt)

            losses_dict['seg_loss'].update(loss_s.item(), input_RGB.size(0))
            if preds_d is not None:
                loss_s.backward(retain_graph=True)
            else:
                loss_s.backward()

        if preds_d is not None:
            loss_d = criterion_dict['criterion_d'](preds_d.squeeze(), depth_gt)
            losses_dict['depth_loss'].update(loss_d.item(), input_RGB.size(0))
            loss_d.backward()

        progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                     ('Seg Loss: %.4f (%.4f) Depth Loss: %.4f (%.4f)'
                      % (losses_dict['seg_loss'].val, losses_dict['seg_loss'].avg,
                         losses_dict['depth_loss'].val, losses_dict['depth_loss'].avg)))
        optimizer.step()

        
    return {'Segmentation Loss': losses_dict['seg_loss'].avg, 'Depth Loss': losses_dict['depth_loss'].avg}


def validate(val_loader, model, num_classes, device, epoch, args, log_dir):
    results_dict = {}
    display_dict = {}
    elapsed_time = 0.0

    with torch.no_grad():
        model.eval()
        if args.do_save_model:
            torch.save(model.state_dict(), os.path.join(log_dir, 'model.ckpt'))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        depth_result_metrics = {'d1': 0.0, 'd2': 0.0, 'd3': 0.0, 'abs_rel': 0.0, 'sq_rel': 0.0,
                                'rmse': 0.0, 'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0}
        tmp_s = []
        tmp_d = []

        for batch_idx, (input_RGB, seg_gt, depth_gt, filename) in enumerate(val_loader):
            seg_gt, depth_gt = seg_gt.long().to(device), depth_gt.to(device)  # B x H x W
            input_RGB = input_RGB.to(device)  # B x 3 x H x W
            start.record()
            preds, preds_d = model(input_RGB)  # B x classes x H x W
            torch.cuda.synchronize()
            end.record()

            if args.is_seg:
                pred_s, seg_gt = preds, seg_gt.squeeze()  # 1 x 1 x H x W
                pred_s = torch.argmax(pred_s, axis=1).squeeze()  # Assume we always validate with batch 1
                confusion_matrix += metrics.get_confusion_matrix(pred_s, seg_gt, num_classes, args.ignore_index)

                if batch_idx < 4:
                    pred_s_numpy = pred_s.cpu().numpy().astype(np.uint8)
                    #input_numpy = input_RGB.cpu().numpy().astype(np.uint8)
                    pred_s_numpy = logging.seg_label_to_color(pred_s_numpy, args.dataset)
                    pred_s_numpy = np.transpose(pred_s_numpy, (2, 0, 1))
                    tmp_s.append(pred_s_numpy)

            if args.is_depth:
                pred_d, depth_gt = preds_d.squeeze(), depth_gt.squeeze()
                computed_result = metrics.eval_depth(pred_d, depth_gt)

                if batch_idx < 4:
                    pred_d_numpy = pred_d.cpu().numpy().astype(np.uint8) * 3
                    # pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_TURBO)
                    pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_BONE)
                    pred_d_color = np.transpose(pred_d_color, (2, 0, 1))
                    tmp_d.append(pred_d_color)

                for key in depth_result_metrics.keys():
                    depth_result_metrics[key] += computed_result[key]

            progress_bar(batch_idx, len(val_loader), args.epochs, epoch)
            elapsed_time += start.elapsed_time(end)

        fps = 1.0 / (elapsed_time / (batch_idx + 1) / 1000)
        print("FPS : ", fps)

        if args.is_seg:
            pixel_acc, mean_acc, mean_iou = metrics.eval_seg(confusion_matrix)
            seg_results_metrics = {'pixel_acc': pixel_acc, 'mean_acc': mean_acc, 'mean_IoU': mean_iou}
            results_dict['Segmentation results'] = seg_results_metrics
            display_dict['pred_s'] = np.stack(tmp_s, axis=0)

        if args.is_depth:
            for key in depth_result_metrics.keys():
                depth_result_metrics[key] = depth_result_metrics[key] / (batch_idx + 1)
            results_dict['Depth results'] = depth_result_metrics
            display_dict['pred_d'] = np.stack(tmp_d, axis=0)

    return display_dict, results_dict, fps


if __name__ == '__main__':
    main()
