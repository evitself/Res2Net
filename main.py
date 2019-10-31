import os
import click
import torch
import cv2
import numpy as np
from tensorflow import keras
from dla import res2net_dla60, res2next_dla60
from res2net import res2net50_14w_8s, res2net50_26w_4s, \
    res2net50_26w_6s, res2net50_26w_8s, res2net50_48w_2s, res2net101_26w_4s
from res2next import res2next50
from pytorch2keras.converter import pytorch_to_keras
from imagenet_1k_labels import imagenet_labels


@click.group()
def cli():
    pass


@cli.command('parse')
@click.option('--model_name', '-n', help='model name')
@click.option('--model_file', '-i', help='input pytorch model file')
@click.option('--keras_model_file', '-o', help='output keras model file')
@click.option('--include_top', '-t', is_flag=True, help='include top block')
@click.option('--variable_size', '-v', is_flag=True, help='variable size')
@click.option('--change_ordering', '-c', is_flag=True, help='change channel ordering')
def main(model_name, model_file, keras_model_file,
         include_top, variable_size, change_ordering):
    print('include top: {}'.format(include_top))
    print('variable size: {}'.format(variable_size))
    print('change ordering: {}'.format(change_ordering))
    rand_tensor = torch.rand(1, 3, 224, 224).cpu()
    model = eval(model_name)(model_file=model_file, include_top=include_top)
    k_model = pytorch_to_keras(model, rand_tensor,
                               [(3, None, None)] if variable_size else [(3, 224, 224)],
                               verbose=False, change_ordering=change_ordering)
    base_dir = os.path.dirname(keras_model_file)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    k_model.save(keras_model_file)


@cli.command()
@click.option('--keras_model', '-m', help='keras model file')
def demo(keras_model):
    model = keras.models.load_model(keras_model)
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        input_tensor = (input_tensor / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        pred = model.predict(np.expand_dims(input_tensor, axis=0))
        pred = pred.squeeze()
        top_id = np.argmax(pred)
        top_score = pred[top_id]
        cv2.putText(frame, '{:3f} - {}'.format(top_score, imagenet_labels[top_id]),
                    (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.imshow('demo', frame)
        key = cv2.waitKey(1)
        if key == ord('x'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    cli()
