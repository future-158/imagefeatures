import json
import os
import pprint
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from matplotlib.dates import date2num, num2date
from omegaconf import OmegaConf


cfg = OmegaConf.load('conf/config.yml')

src_dir = Path(cfg.catalogue.image_dir)
out_dir = Path(cfg.catalogue.out_dir)
label_path = Path(cfg.catalogue.label_path)


files = list(src_dir.glob('**/*.jpg'))

table = pd.read_csv(label_path, parse_dates=['datetime'])

file_df = pd.DataFrame(data = dict(
    stem = [x.stem for x in files],
    fullname = [x.as_posix() for x in files]
))

table = pd.merge(file_df, table, how='inner', on = ['stem'])
table['fdatetime'] = date2num(table.datetime)

num_files = len(table)
print('num files is: ', num_files)
num_samples = 1024
num_tfrecords = num_files // num_samples

if num_files % num_samples:
    num_tfrecords += 1  # add one record if there are any remaining samples

if not os.path.exists(out_dir):
    os.makedirs(out_dir)  # creating TFRecords output folder

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


@dataclass
class Annot():
    label: int
    split: str
    fdatetime: float
    stem: str

def create_example(image, annot: Annot):
    feature = {
        "image": image_feature(image),
        "label": int64_feature(annot.label),
        "split": bytes_feature(annot.split),
        "fdatetime": float_feature(annot.fdatetime),
        "stem": bytes_feature(annot.stem),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
    
def parse_tfrecord_fn(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "split": tf.io.FixedLenFeature([], tf.string),
        "fdatetime": tf.io.FixedLenFeature([], tf.float32),
        "stem": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
    return example


for tfrec_num in range(num_tfrecords):
    samples = table.iloc[tfrec_num* num_samples : (tfrec_num+1)*num_samples]


    with tf.io.TFRecordWriter(
        out_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
    ) as writer:
        for sample in samples.itertuples():
            a = Annot(
                label = sample.label,
                split = sample.split,
                fdatetime = datetime.timestamp(sample.datetime.to_pydatetime()), #return differently
                stem = sample.stem

            )
    
            image_path = sample.fullname
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            example = create_example(image, a)
            writer.write(example.SerializeToString())
