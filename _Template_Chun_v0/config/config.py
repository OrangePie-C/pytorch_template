import argparse
import yaml


config_dict = yaml.load(f, Loader=yaml.FullLoader)



def Load_Config():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='gpu')
    parser = parser.parse_args()

    config_dict = yaml.load(parser.config, Loader=yaml.FullLoader)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for i in config_dict:
        parser.add_argument(i, type=type(config_dict[i]), default=config_dict[i])
        # if isinstance(config_dict[i], str):
        #     parser.add_argument(i, type=str, default=config_dict[i])
        #     pass
        # elif isinstance(config_dict[i], int):
        #     parser.add_argument(i, type=int, default=config_dict[i])
        # else:
        #     exit("Invalid Config")

def Train_Config():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # base configs
    parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
    parser.add_argument('--data_path',    type=str, default='/mnt/ssd1/')
    parser.add_argument('--dataset',      type=str, default='cityscapes',
                        choices=['cityscapes', 'nyudepthv2', 'kitti', 'ImagePath'])

    return parser

def Test_Config():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # base configs
    parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
    parser.add_argument('--data_path',    type=str, default='/mnt/ssd1/')
    parser.add_argument('--dataset',      type=str, default='cityscapes',
                        choices=['cityscapes', 'nyudepthv2', 'kitti', 'ImagePath'])


    return parser