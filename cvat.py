# Loads CVAT XML annotations and generates binary masks for each image
# Outputs masks to directory "cvat_masks"

import os
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Argument: filename (without extension) of the CVAT XML file to load (e.g. "shadow2" for "shadow2.xml")
def load_image(filename):
    outdir = os.path.join("cvat_masks", f"{filename}_mask.png")
    if os.path.isfile(outdir):
        print("Benchmark image already exists")
        return cv2.imread(outdir, cv2.IMREAD_UNCHANGED)
    else:
        xml_path = os.path.join("xml", f"{filename}.xml")
        if not os.path.isfile(xml_path):
            return None
        print("Benchmark image does not exist, generating...")
        root = ET.parse(xml_path).getroot()
        for image in root.iter("image"):
            name = os.path.splitext(image.get("name"))[0]
            h, w = int(image.get("height")), int(image.get("width"))
            mask = np.zeros((h, w), np.uint8)
            for poly in image.iter("polygon"):
                pts = [tuple(map(float, p.split(","))) for p in poly.get("points").split(";")]
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
            cv2.imwrite(outdir, mask)
            return mask
        