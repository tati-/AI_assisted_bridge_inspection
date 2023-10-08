"""
This script tests an onnx model with a single image, or a number of images found
in a folder.
It then plots and saves the original image, next to the
prediction
example usage: python test_onnx.py -modelpath path/to/model.onnx -cl path/to/class_description
"""
import os
import sys
import pdb
import cv2
import math
import argparse
import onnxruntime
import numpy as np
from pathlib import Path
from natsort import natsorted

sys.path.append('..')
import modules.utils as utils
import modules.visualization as vis
from modules.semantic_categories import labels_pipo

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-imgpath', help='test image path, can be folder or file [default: ../../data/test_img.png]', type=str, nargs='+', default='../../data/test_img.png')
    arg('-modelpath', help='trained onnx model path', required=True, type=str)

    args = parser.parse_args()
    return args


def onnx_predict(onnx_model: onnxruntime.InferenceSession, x_test):
    """
    this function uses an onnx model to predict on a number of images
    and returns the prediction array
    INPUTS:
    @onnx_model: onnx runtime inference session
    @x_test: array in the form of [nSamples, height, width, nChanels]
    """
    output_names = [x.name for x in onnx_model.get_outputs()]
    input_names = [x.name for x in onnx_model.get_inputs()]
    # prepare input
    # we assume for now the network has a single input
    height, width = onnx_model.get_inputs()[0].shape[1:3]
    x_test = np.asarray([cv2.resize(x, (width, height)) for x in x_test])
    x_test = x_test.astype('float32') # onnx model needs float32
    # plt.imsave('test_img_resized.png', x_test[0])

    # onnx model inference
    onnx_pred = []
    # for the moment assumes there is a single input to the model
    batch_size = 10
    for batch_ind in range(0, len(x_test), batch_size):
        onnx_pred.extend(onnx_model.run(output_names,
            {input_names[0]: x_test[batch_ind:min(batch_ind+batch_size, len(x_test))]})[0])
    return np.asarray(onnx_pred)


if __name__ == "__main__":
    args = get_args()
    # check input validity
    assert Path(args.modelpath).suffix == '.onnx', 'Onnx format model required as input'

    # load onnx model for inference
    # onnx_model = onnx.load("input_path")  # load onnx model
    # output = prepare(onnx_model).run(input)  # run the loaded model
    onnx_model = onnxruntime.InferenceSession(args.modelpath, None)

    # load test image(s)
    if os.path.isdir(args.imgpath[0]):
        test_img_paths = utils.files_with_extensions('jpg', 'JPG', 'png', datapath=args.imgpath)
    else:
        test_img_paths = args.imgpath

    # process images in batches of 500 max, otherwise crash
    for i in range(math.ceil(len(test_img_paths)/500)):
        start, end = i*500, min(i*500+500, len(test_img_paths))
        # test_imgs = list()
        # for x in test_img_paths:
        #     try:
        #         test_imgs.append(cv2.cvtColor(cv2.imread(x),  cv2.COLOR_BGR2RGB))
        #     except:
        #         continue
        # test_imgs = np.asarray(test_imgs)
        test_imgs = np.asarray([cv2.cvtColor(cv2.imread(x),  cv2.COLOR_BGR2RGB) for x in test_img_paths[start:end]])
        if np.max(test_imgs)>100 and 'rescale' not in args.modelpath:
            test_imgs = test_imgs/255;
        onnx_preds = onnx_predict(onnx_model, test_imgs)
        onnx_preds = onnx_preds.astype(int)
        # np.savetxt('onnx_preds.csv', onnx_preds[0,:], fmt='%d', delimiter=',')

        # visualize results
        savefolder = os.path.join(Path(args.imgpath[0]).parent, 'onnx_results',
                                  Path(args.modelpath).stem)
        os.makedirs(savefolder, exist_ok=True)
        vis.inspect_predictions(*[(test_imgs[i], None, onnx_preds[i]) for i in range(len(test_imgs))],
                                labels=labels_pipo, savefolder=savefolder)
        print(f'Inspection predictions saved in {savefolder}')
"""
############################################################################
                                END
############################################################################
"""
