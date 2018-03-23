import mxnet as mx
import argparse

parser = argparse.ArgumentParser(description='draw CNN network from json file')
parser.add_argument('--json', type=str,help='the input json file')
args = parser.parse_args()
sy = mx.symbol.load(args.json)
a = mx.viz.plot_network(sy)
a.view()