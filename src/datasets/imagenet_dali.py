"""ImageNet.
    Define two class named HybridTrainPipe and HybridValPipe, since data
    augementation is different.
"""

import os
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from\
                      https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads,
                                              device_id, seed=12+device_id)
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    random_shuffle=True)
        # let user decide which pipeline works him bets for RN version he runs.
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able
        # to handle all images from full-sized ImageNet without additional
        # reallocation.
        device_memory_padding = 211025920 if decoder_device == "mixed" else 0
        host_memory_padding = 140544512 if decoder_device == "mixed" else 0
        self.decode = ops.ImageDecoderRandomCrop(
            device=decoder_device, output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            random_aspect_ratio=[0.8, 1.25], random_area=[0.1, 1.0],
            num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255,
                                                  0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255,
                                                 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id,
                                            seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255,
                                                  0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255,
                                                 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(mode, data_dir, batch_size, num_threads,
                           device_id=0, num_gpus=1, crop=224,
                           val_size=256, world_size=1, local_rank=0):
    data_dir = os.path.join(data_dir, "ilsvrc2012")
    if mode == "train":
        data_dir = os.path.join(data_dir, "train")
        pip_train = HybridTrainPipe(batch_size=batch_size,
                                    num_threads=num_threads,
                                    device_id=local_rank, data_dir=data_dir,
                                    crop=crop, world_size=world_size,
                                    local_rank=local_rank)
        pip_train.build()
        size = int(pip_train.epoch_size("Reader") / world_size)
        dali_iter_train = DALIClassificationIterator(pip_train, size=size)
        return dali_iter_train
    elif mode == "val":
        data_dir = os.path.join(data_dir, "val")
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads,
                                device_id=local_rank, data_dir=data_dir,
                                crop=crop, size=val_size,
                                world_size=world_size, local_rank=local_rank)
        pip_val.build()
        size = int(pip_val.epoch_size("Reader") / world_size)
        dali_iter_val = DALIClassificationIterator(pip_val, size=size)
        return dali_iter_val


if __name__ == "__main__":
    import time
    data_dir = "/media/lincolnzjx/HardDisk/Datasets/"
    # data_dir = "/data15/Public/Datasets/"
    train_loader = get_imagenet_iter_dali(mode="train", data_dir=data_dir,
                                          batch_size=256, num_threads=4,
                                          crop=224, device_id=0, num_gpus=1)
    print("Start Iterate On ImageNet Train Dataset")
    start = time.time()
    for index, data in enumerate(train_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    end = time.time()
    print("End Iterate, Dali iterate time: {.f}s".format(end-start))

    val_loader = get_imagenet_iter_dali(mode="val", data_dir=data_dir,
                                        batch_size=256, num_threads=4,
                                        crop=224, device_id=0, num_gpus=1)
    print("Start Iterate On ImageNet Val Dataset")
    start = time.time()
    for index, data in enumerate(val_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    end = time.time()
    print("End Iterate, Dali iterate time: {.f}s".format(end-start))
