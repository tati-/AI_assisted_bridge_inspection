"""
This script creates an onnx model from a tensorflow keras model.
example usage: python tfKeras2onnx.py -modelpath path/to/SavedModel
"""
import os
import sys
import pdb
import cv2
import warnings
import argparse
import numpy as np
from pathlib import Path

sys.path.append('..')
from modules.decorators import timer
from modules.visualization import tfonnx_comparison
exec(open('../modules/ml_imports.py').read())

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-modelpath', help='trained onnx model path', required=True, type=str)
    arg('--onehot', help='if set the network output is one hot encoded', action='store_true')
    arg('--rescale', help='if set the network input is devided by 255 (if the input is 0-255)', action='store_true')
    arg('--resize', help='if set the input is resized to match the network input size', action='store_true')
    arg('-img', help='test image path to check the consistency between onnx and tensorflow [default: ../../data/test_img.png]',
                type=str, default='../../data/test_img.png')

    args = parser.parse_args()
    return args


@timer
def add_processing_layers(modelpath, onehot=False, rescale=False, resize=False, **kwargs):
    """
    take a tensorfow model path and potentially add some pre and post processing
    layers so that it can then be converted in an onnx model
    INPUTS:
    @modelpath: the path to the tensorflow model (saved model format)
    @onehot: if set the output of the model is one hot (1 for the predicted class, 0 for all others)
                otherwise it will be categorical (integer in [1, nClasses])
    @rescale: If set a layer is added to the onnx model before the input that divides the
                input values by 255
    @resize: If set a layer is added to the onnx model before the input that resizes the
                input image to the model input size
    """
    # load model and add potential additional layers
    model = tf.keras.models.load_model(modelpath, compile=False)

    # reexamine this later, maybe I need to do this at training level
    # input = Input(shape=(None, None, 3)) if args.resize else model.input
    # this does not work as well, graph disconnected
    # input = Input(shape=(500, 520, 3)) if args.resize else model.input
    if resize:
        prepro = Resizing(height=model.input.shape[1], width=model.input.shape[2])(model.input)
    if rescale:
        prepro = Rescaling(scale=1./255)(prepro) if args.resize else Rescaling(scale=1./255)(model.input)
    if resize or rescale:
        rest_model = Model(inputs=model.layers[1].input, outputs = model.output)
        output = rest_model(prepro)
        model = Model(inputs=[model.input], outputs=[output])
    mask = tf.math.argmax(model.output, axis=-1, name='mask')
    if onehot:
        one_hot = tf.one_hot(mask, model.output.shape[-1])
        model = Model(inputs=[model.input], outputs=[one_hot])
    else:
        model = Model(inputs=[model.input], outputs=[mask])
    model.summary()

    return model


def sanity_check_onnx(tf_model, onnx_model, test_img):
    """
    this functions verifies that the output of an onnx model is the same as that
    of a tensorflow model
    INPUTS:
    @tf_model: an instance of a tensorflow model
    @onnx_model: an instance of a tensorflow model
    """
    # if there is a single test image, a 1 element array is created to be passed
    # in the models. This also allowes for the extension of this function in the
    # future to test with multiple images
    x_test = np.asarray([test_img]) if test_img.ndim==3 else test_img
    if x_test.max()>2:
        x_test = x_test/255
    output_names = [x.name for x in onnx_model.get_outputs()]
    # resize test_img if need be
    height, width = onnx_model.get_inputs()[0].shape[1:3]
    x_test = np.asarray([cv2.resize(x, (width, height)) for x in x_test])
    x_float = x_test.astype('float32') # onnx model needs float32

    onnx_preds = np.asarray(onnx_model.run(output_names, {tf_model.input.name: x_float})[0])
    tf_preds = tf_model.predict(x_test)
    if tf_preds.ndim==4:
        tf_preds = np.argmax(np.squeeze(tf_preds), axis=-1)
        onnx_preds = np.argmax(np.squeeze(onnx_preds), axis=-1)

    # if np.allclose(tf_preds, onnx_preds):
    if np.sum(tf_preds!=onnx_preds)/tf_preds.size<0.01:
        print(f'{"#"*50}\nSanity check passed, tf and onnx behave similarly\n{"#"*50}')
    else:
        #visualize
        savepath = './tf_onnx_compare.png'
        tfonnx_comparison(x_test[0], tf_preds[0], onnx_preds[0], savepath)
        print(f'{"x"*50}\nThe tf and ONNX models do not produce the same output!\n'\
            f'{np.sum(tf_preds!=onnx_preds)/tf_preds.size*100:.3f}% pixels differ\n'\
            f'Check image {savepath} to see if the difference is significant\n{"x"*50}')


if __name__ == "__main__":
    if len(tf.config.list_physical_devices('GPU'))==0:
        print('#'*10)
        print('NO GPU IS USED!!!')
        print('#'*10)
        # Hide GPU from visible devices
        # tf.config.sett_visible_devices([], 'GPU')
    args = get_args()
    # check input validity
    assert os.path.isdir(args.modelpath), 'Model should be in TensorFlow SavedModels format'
    ## ++ here check also contents of modepath folder

    # add potential pre and post processing layers
    tf_model = add_processing_layers(**vars(args))

    # define onnx model naming based on arguments
    if args.onehot:
        suffix = '_onehot'
    else:
        suffix = '_categorical'
    if args.rescale:
        suffix += '_rescale'
    if args.resize:
        suffix += '_resize'

    # save model in onnx format
    onnx_path = os.path.join(Path(args.modelpath).parent,
                         f'{Path(args.modelpath).name}{suffix}.onnx')
    tf2onnx.convert.from_keras(tf_model, output_path=onnx_path)
    print(f'Onnx model saved in {onnx_path}')

    # validate that onnx and tensorflow outputs are the same on a test image
    onnx_model = onnxruntime.InferenceSession(onnx_path, None)
    test_img = cv2.cvtColor(cv2.imread(args.img),  cv2.COLOR_BGR2RGB)
    sanity_check_onnx(tf_model, onnx_model, test_img)
"""
############################################################################
                                END
############################################################################
"""
    #####
    # # with input_signature - no difference in the output
    # pdb.set_trace()
    # print('With input signature: \n')
    # spec = (tf.TensorSpec(tuple(tf_model.input.shape), tf.float32, name="input"),)
    # tf2onnx.convert.from_keras(tf_model, input_signature=spec, output_path='test_input_signature.onnx')
    # onnx_model = onnxruntime.InferenceSession('test_input_signature.onnx', None)
    # sanity_check_onnx(tf_model, onnx_model, test_img)
    #####
    # # with specified inputs as nchw - this creates an error as it expects an input
    # # already in nchw format, so it ommits the transpose
    # pdb.set_trace()
    # print('With specified inputs as nchw: \n')
    # tf2onnx.convert.from_keras(tf_model, inputs_as_nchw=[x.name for x in tf_model.inputs], output_path='test_inputs_as_nchw.onnx')
    # onnx_model = onnxruntime.InferenceSession('test_inputs_as_nchw.onnx', None)
    # sanity_check_onnx(tf_model, onnx_model, test_img)
