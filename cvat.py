# Loads CVAT XML annotations and generates binary masks for each image
# Outputs masks to directory "cvat_masks"

import os
import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Argument: filename (without extension) of the CVAT XML file to load (e.g. "shadow2" for "shadow2.xml")
def load_image(filename):
    xml_path = os.path.join("xml", f"{filename}.xml")
    root = ET.parse(xml_path).getroot()
    for image in root.iter("image"):
        name = os.path.splitext(image.get("name"))[0]
        h, w = int(image.get("height")), int(image.get("width"))
        mask = np.zeros((h, w), np.uint8)
        for poly in image.iter("polygon"):
            pts = [tuple(map(float, p.split(","))) for p in poly.get("points").split(";")]
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
        out = os.path.join("cvat_masks", f"{name}_mask.png")
        cv2.imwrite(out, mask)
        print("Saved", out)