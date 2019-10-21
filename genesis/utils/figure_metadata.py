import sys

import matplotlib.pyplot as plt
from PIL import Image


def patch_savefig_for_argv_metadata():
    savefig_old = plt.savefig
    def savefig(*args, **kwargs):
        savefig_old(metadata=dict(argv=" ".join(sys.argv)), *args, **kwargs)
    plt.savefig = savefig

def print_fig_metadata(filename):
    img = Image.open(filename)
    print("file generated with:")
    print()
    print(img.info['argv'])

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename')

    args = argparser.parse_args()

    print_fig_metadata(args.filename)
