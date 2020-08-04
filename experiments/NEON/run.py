import argparse
from DeepTreeAttention import species

def parse_args():
    """Set training mode hrough command line"""
    parser = argparse.ArgumentParser(description='NEON species prediction pipeline')
    parser.add_argument('train', action='store_true')
    args = parser.parse_args()
    
    return args

if __name == "__main__":
    args = parse_args()
    species.main(dirname="/orange/idtrees-collab/predictions/", generate=True, cpus=2, train=args.train)
