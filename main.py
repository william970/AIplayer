import argparse
from models.model import Resnet
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='dataSet')
    parser.add_argument('--weights', type=str, default='weights/best.pt')
    parser.add_argument('--config', type=str, default='transformer')
    resnet = Resnet()