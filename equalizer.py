import argparse
import cv2
# construct the argument parser and parse the arguments

def equalizer(gray):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=False,
        help="path to the input image")
    ap.add_argument("-c", "--clip", type=float, default=2.0,
        help="threshold for contrast limiting")
    ap.add_argument("-t", "--tile", type=int, default=8,
        help="tile grid size -- divides image into tile x time cells")
    args = vars(ap.parse_args())

 
    clahe = cv2.createCLAHE(clipLimit=args["clip"],
        tileGridSize=(args["tile"], args["tile"]))
    equalized = clahe.apply(gray)
    return equalized