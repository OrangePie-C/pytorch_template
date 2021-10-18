import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# base configs
parser.add_argument('-c', '--config', type=str, default='gpu')

parser = parser.parse_args()
print(parser.config)