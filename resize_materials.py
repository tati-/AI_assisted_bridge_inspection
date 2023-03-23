"""
This scripts takes a file containing textures and limits all the images' width to
a maximun number of pixels/
# sample usage: python resize_materials.py -m ~/Desktop/bridge_materials_mid_res/bridge_materials.blend
"""
import os
import pdb
import bpy
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-m', '-materials', help='folder containing the blender file with the materials', type=str,  required=True)
    arg('-size', help='size of biggest image dimension', type=int, default=1024)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.m))
    for img in tqdm(bpy.data.images, desc='Processing images:'):
        try:
            ratio = img.size[0] / img.size[1]
        except:
            continue
        img.scale(args.size, int(args.size/ratio))
        img.save()
        img.pack()
    bpy.ops.wm.save_as_mainfile()
    """
    ############################################################################
                                    END
    ############################################################################
    """
