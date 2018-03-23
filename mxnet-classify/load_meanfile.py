# -- coding utf-8 --
# Time: 18/03/23
__author__ = 'Yawei Li'

# issue: computing image mean and How to get or generate the mean_image?
# links:

url_01 = 'http://mxnet.incubator.apache.org/api/python/io/io.html#module-mxnet.recordio' #Data Loading API
url_02 = 'https://github.com/apache/incubator-mxnet/issues/604'

add_mean_img = "../recordIO_dir/mean.bin"

# data
def get_iterator(args, kv):
    data_shape = (3, args.data_shape, args.data_shape)
    train = mx.io.ImageRecordIter(
        path_imgrec = os.path.join(args.data_dir, args.train_dataset),
        mean_img = "mean.bin",  # mean_image
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        num_parts   = kv.num_workers,
        part_index  = kv.rank)