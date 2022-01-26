import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_eager_execution()

def parse_example(example):
    expected_features = {}
    expected_features['Xi'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['Xv'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    expected_features['labels'] = tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    parsed_feature_dict = tf.parse_single_example(example, features=expected_features)
    label = parsed_feature_dict['labels']

    label = tf.io.decode_raw(label, out_type=tf.float32)
    label = tf.reshape(label, [])
    Xi = tf.io.decode_raw(parsed_feature_dict['Xi'], out_type=tf.float32)
    Xi = tf.reshape(Xi, [13])
    Xv = tf.io.decode_raw(parsed_feature_dict['Xv'], out_type=tf.float32)
    Xv = tf.reshape(Xv, [13])
    parsed_feature_dict['Xi'] = Xi
    parsed_feature_dict['Xv'] = Xv
    parsed_feature_dict.pop('labels')

    return parsed_feature_dict, label

reader = tf.TFRecordReader()
filename = '/Users/chendi/PycharmProjects/kg_experimental_thesis/data/tfrecords_dataset/valid/0/2021_12_06_19_33_07.tfrecords'

ds = tf.data.TFRecordDataset(filename)
ds = ds.map(lambda x: parse_example(x)).prefetch(buffer_size=10).batch(10)
itr = ds.make_one_shot_iterator()
batch_data = itr.get_next()
res = tf.Session().run(batch_data)

print(res)