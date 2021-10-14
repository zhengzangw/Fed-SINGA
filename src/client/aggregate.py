# save load states constant
TENSOR_DICT_FILENAME = '/tensor_dict.npz'
STATES_ATTR_FILENAME = '/states_attr.json'
MODEL_STATE_TYPE = 0
AUX_STATE_TYPE = 1
CENTRAL_MODEL_PATH = "checkpoint/central_model.zip"

import os
import time
import zipfile
import json
import numpy as np
from singa import tensor
from singa import device
from singa import opt

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}

# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


def load_states(fpath):
    assert os.path.isfile(fpath), (
        "Failed to load states, %s is not exist." % fpath)

    timestamp = time.time()
    tmp_dir = '/tmp/singa_load_states_%s' % timestamp
    os.mkdir(tmp_dir)

    with zipfile.ZipFile(fpath, 'r') as zf:
        zf.extractall(tmp_dir)

    tensor_dict_fp = tmp_dir + TENSOR_DICT_FILENAME
    states_attr_fp = tmp_dir + STATES_ATTR_FILENAME

    with open(states_attr_fp) as f:
        states_attr = json.load(f)

    tensor_dict = np.load(tensor_dict_fp)

    # restore singa tensor from numpy
    model_states = dict()
    aux_states = dict()

    for k in tensor_dict.files:
        if states_attr[k]['state_type'] == MODEL_STATE_TYPE:
            model_states[k] = tensor.from_numpy(tensor_dict[k])
        elif states_attr[k]['state_type'] == AUX_STATE_TYPE:
            aux_states[k] = tensor.from_numpy(tensor_dict[k])

    # clean up tmp files
    os.remove(tensor_dict_fp)
    os.remove(states_attr_fp)
    os.rmdir(tmp_dir)
    return model_states, aux_states
    
def save_states(fpath, states, aux_states={}):
    """Save states.
    Args:
        fpath: output file path (without the extension)
        aux_states(dict): values are standard data types or Tensor,
                          e.g., epoch ID, learning rate, optimizer states
    """
    assert not os.path.isfile(fpath), (
        "Failed to save states, %s is already existed." % fpath)


    # save states data and attr
    tensor_dict = {}
    states_attr = {}
    for k, v in states.items():
        assert isinstance(v, tensor.Tensor), "Only tensor state is allowed"
        tensor_dict[k] = tensor.to_numpy(v)
        states_attr[k] = {
            'state_type': MODEL_STATE_TYPE,
            'shape': v.shape,
            'dtype': v.dtype
        }

    for k, v in aux_states.items():
        assert isinstance(v,
                          tensor.Tensor), "Only tensor aux state is allowed"
        tensor_dict[k] = tensor.to_numpy(v)
        states_attr[k] = {
            'state_type': AUX_STATE_TYPE,
            'shape': v.shape,
            'dtype': v.dtype
        }

    # save to files
    timestamp = time.time()
    tmp_dir = '/tmp/singa_save_states_%s' % timestamp
    os.mkdir(tmp_dir)
    tensor_dict_fp = tmp_dir + TENSOR_DICT_FILENAME
    states_attr_fp = tmp_dir + STATES_ATTR_FILENAME

    np.savez(tensor_dict_fp, **tensor_dict)

    with open(states_attr_fp, 'w') as fp:
        json.dump(states_attr, fp)

    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(fpath, mode="w") as zf:
        zf.write(tensor_dict_fp,
                 os.path.basename(tensor_dict_fp),
                 compress_type=compression)
        zf.write(states_attr_fp,
                 os.path.basename(states_attr_fp),
                 compress_type=compression)

    # clean up tmp files
    os.remove(tensor_dict_fp)
    os.remove(states_attr_fp)
    os.rmdir(tmp_dir)

def eval():
    dev=device.get_default_device()
    batch_size = 64
    precision='float32'
    world_size = 1

    from data import mnist
    train_x, train_y, val_x, val_y = mnist.load()
    num_channels = train_x.shape[1]
    image_size = train_x.shape[2]
    data_size = np.prod(train_x.shape[1:train_x.ndim]).item()
    num_classes = (np.max(train_y) + 1).item()

    from model import cnn
    model = cnn.create_model(num_channels=num_channels, num_classes=num_classes)

    if model.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, num_channels, model.input_size, model.input_size), dev,
            singa_dtype[precision])
    elif model.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev, singa_dtype[precision])
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))
    
    # Attach model to graph
    sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype[precision])
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=True, sequential=False)
    dev.SetVerbosity(False)
    # print("loading model from " + CENTRAL_MODEL_PATH)
    model.load_states(fpath=CENTRAL_MODEL_PATH	)
    
    # print(model.get_states()['linear2.b'])

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_val_batch = val_x.shape[0] // batch_size
    test_correct = np.zeros(shape=[1], dtype=np.float32)
    model.eval()

    
    for b in range(num_val_batch):
        x = val_x[b * batch_size:(b + 1) * batch_size]
        if model.dimension == 4:
            if (image_size != model.input_size):
                x = resize_dataset(x, model.input_size)
        x = x.astype(np_dtype[precision])
        y = val_y[b * batch_size:(b + 1) * batch_size]
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)
        out_test = model(tx)
        test_correct += accuracy(tensor.to_numpy(out_test), y)


    # Output the evaluation accuracy
    print('Evaluation accuracy = %f' %
          (test_correct / (num_val_batch * batch_size * world_size)))


def aggregate(num):
    model_states = {}
    for d in range(num):
        checkpointpath="checkpoint/checkpoint_" + str(d) + ".zip"
        tmp, aux_states = load_states(checkpointpath)
        if not bool(model_states):
            model_states = tmp
        else:
            for k, v in tmp.items():
                model_states[k] = model_states[k] + v
    
    for k, v in model_states.items():
        model_states[k] = model_states[k]/num

    if os.path.exists(CENTRAL_MODEL_PATH):
        os.remove(CENTRAL_MODEL_PATH)
    save_states(CENTRAL_MODEL_PATH, model_states)
    

aggregate(10)
eval()
