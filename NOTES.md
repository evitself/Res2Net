# Notes for converting

We are using pytorch2keras package to convert models to keras. The underlying dependency is onnx conversion.
There are 2 steps:

- step 1), convert pytorch model to onnx
- step 2), second, use onnx2keras to convert onnx model to keras model

The key issue here is the `split` operation. In res2net, it splits tensor along the `channel` axis.
In pytorch we use `NCHW` order by default, which is `channel first`. So, if you convert to keras with `change ordering`, which convert tensor format to `channel last`, you will have to handle `split` operation conversion specifically.

Let's see `<your python folder>/site-packages/onnx2keras/operation_layers.py`

Find function 

```python

def convert_split(node, params, layers, node_name, keras_names):
    """
    Convert Split layer
    :param node: current operation node
    :param params: operation attributes
    :param layers: available keras layers
    :param node_name: internal converter name
    :param keras_name: resulting layer name
    :return: None
    """
    if len(node.input) != 1:
        assert AttributeError('More than 1 input for split layer.')

    input_0 = ensure_tf_type(layers[node.input[0]], name="%s_const" % keras_names[0])
    splits = params["split"]
    axis = params.get("axis", 0)
    if not isinstance(splits, Iterable):
        # This might not work if `split` is a tensor.
        chunk_size = K.int_size(input_0)[axis] // splits
        splits = (chunk_size,) * splits

    cur = 0
    for i, split in enumerate(splits):
        node_name = params['_outputs'][i]

        def target_layer(x, axis=axis, start_i=cur, end_i=cur+split):
            slices = [slice(None, None)] * len(K.int_shape(x))
            slices[axis] = slice(start_i, end_i)
            return x[tuple(slices)]

        lambda_layer = keras.layers.Lambda(target_layer, name=keras_names[i])
        layers[node_name] = lambda_layer(input_0)
        cur += split
``` 

It converts `split` operation to a keras `lambda` layer

Then let's take a look how keras serializes `lambda` layer to a config object

Find keras `lambda` layer source code here: `tensorflow/python/keras/layers/core.py`

```python
  def _serialize_function_to_config(self, inputs, allow_raw=False):
    if isinstance(inputs, python_types.LambdaType):
      output = generic_utils.func_dump(inputs)
      output_type = 'lambda'
      module = inputs.__module__
    elif callable(inputs):
      output = inputs.__name__
      output_type = 'function'
      module = inputs.__module__
    elif allow_raw:
      output = inputs
      output_type = 'raw'
      module = None
    else:
      raise ValueError(
          'Invalid input for serialization, type: %s ' % type(inputs))
```

Good, `func_dump` is the key.

Navigate to `tensorflow/python/keras/utils/generic_utils.py`

```python
def func_dump(func):
  """Serializes a user defined function.
  Arguments:
      func: the function to serialize.
  Returns:
      A tuple `(code, defaults, closure)`.
  """
  if os.name == 'nt':
    raw_code = marshal.dumps(func.__code__).replace(b'\\', b'/')
    code = codecs.encode(raw_code, 'base64').decode('ascii')
  else:
    raw_code = marshal.dumps(func.__code__)
    code = codecs.encode(raw_code, 'base64').decode('ascii')
  defaults = func.__defaults__
  if func.__closure__:
    closure = tuple(c.cell_contents for c in func.__closure__)
  else:
    closure = None
  return code, defaults, closure
```

So, for serialized `lambda` layer, first element is serialized byte code, second one is defaults for variables, third is closure.
Let's go back to `convert_split` function:

```python
        def target_layer(x, axis=axis, start_i=cur, end_i=cur+split):
            slices = [slice(None, None)] * len(K.int_shape(x))
            slices[axis] = slice(start_i, end_i)
            return x[tuple(slices)]
```

There should be 3 variables, in the order of appearance, they are `axis, start_i, end_i`. 
`axis` is the one to need to update:
- `1` means channel first, that is channel 1 in `NCHW`
- `3` means channel last, that is channel 3 in `NHWC`

Now we know how to handle the logic correctly, navigate to file `<your python folder>/site-packages/onnx2keras/converter.py`
Find function `onnx_to_keras`, in the `if change_ordering:` block, change source code as following:

```python
    if change_ordering:
        import numpy as np
        conf = model.get_config()

        for layer in conf['layers']:
            if layer['config'] and 'batch_input_shape' in layer['config']:
                layer['config']['batch_input_shape'] = \
                    tuple(np.reshape(np.array(
                        [
                            [None] +
                            list(layer['config']['batch_input_shape'][2:][:]) +
                            [layer['config']['batch_input_shape'][1]]
                        ]), -1
                    ))
            if layer['config'] and 'target_shape' in layer['config']:
                if len(list(layer['config']['target_shape'][1:][:])) > 0:
                    layer['config']['target_shape'] = \
                        tuple(np.reshape(np.array(
                            [
                                list(layer['config']['target_shape'][1:][:]),
                                layer['config']['target_shape'][0]
                            ]), -1
                        ), )

            if layer['config'] and 'data_format' in layer['config']:
                layer['config']['data_format'] = 'channels_last'
            if layer['config'] and 'axis' in layer['config']:
                layer['config']['axis'] = 3

        for layer in conf['layers']:
            if 'function' in layer['config'] and layer['config']['function'][1] is not None:
                # Modification starts here:
                f = list(layer['config']['function'])
                split_config = layer['config']['function'][1]
                if split_config[0] == 1:
                    print('current ordering is channel first, needs to be converted')
                    split_config = (3, split_config[1], split_config[2])
                    layer['config']['function'] = tuple([layer['config']['function'][0],
                                                         split_config,
                                                         layer['config']['function'][2]])

                # ORIGINAL CODE:
                # if len(layer['config']['function'][1][0].shape) == 4:
                #     f[1] = (np.transpose(layer['config']['function'][1][0], [0, 2, 3, 1]), f[1][1])
                # elif len(layer['config']['function'][1][0].shape) == 3:
                #     f[1] = (np.transpose(layer['config']['function'][1][0], [0, 2, 1]), f[1][1])
                # layer['config']['function'] = tuple(f)

        keras.backend.set_image_data_format('channels_last')
        model_tf_ordering = keras.models.Model.from_config(conf)

        for dst_layer, src_layer in zip(model_tf_ordering.layers, model.layers):
            dst_layer.set_weights(src_layer.get_weights())

        model = model_tf_ordering

    return model

```

Then you should be able to convert res2net model correctly
