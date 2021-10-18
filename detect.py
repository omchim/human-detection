import argparse
import torch
from core import HumanDetection
import time

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, type=str,
        help="path to input image")
    ap.add_argument("-s", "--skip", required=True, type=int, default=20,
        help="Skip frame detection frequence")

    args = ap.parse_args()


    hd = HumanDetection(args.image, args.skip)
    start_time = time.time()
    hd.detect()
    print("--- %s seconds ---" % (time.time() - start_time))
