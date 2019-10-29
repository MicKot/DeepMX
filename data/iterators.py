import mxnet as mx
import random


class ImageIter(mx.io.DataIter):
    def __init__(
        self,
        path=None,
        rec_file=False,
        batch_size=32,
        shuffle=True,
        number_of_labels=1,
        label_preprocess_func=None,
        image_preprocess_func=None,
        output_size=(3, 224, 224),
    ):
        super(ImageIter, self).__init__()
        self.current_batch = 0
        self.current_index = 0
        self.number_of_labels = number_of_labels
        self.batch_size = batch_size
        self.path = path
        self.rec_file = rec_file
        self.shuffle = shuffle
        self.label_preprocess_func = label_preprocess_func
        self.image_preprocess_func = image_preprocess_func
        self.output_size = output_size
        self._provide_data = list(zip(["data"], [(self.batch_size, *self.output_size)]))
        self._provide_label = list(
            zip(
                ["output_" + str(x) for x in range(self.number_of_labels)],
                [(self.batch_size,) for x in range(self.number_of_labels)],
            )
        )
        if self.rec_file:
            self.record_file = mx.recordio.MXIndexedRecordIO(
                self.path + ".idx", self.path + ".rec", "r"
            )
            self.number_of_batches = len(self.record_file.keys) // self.batch_size
            self.image_indices = list(self.record_file.keys)
            if self.shuffle:
                random.shuffle(self.image_indices)
        else:
            raise "Not implemented"

    def __iter__(self):
        return self

    def __len__(self):
        return self.number_of_batches

    def reset(self):
        self.current_index = 0
        self.current_batch = 0
        if self.shuffle:
            random.shuffle(self.image_indices)

    def next_sample(self):
        sample = self.record_file.read_idx(self.image_indices[self.current_index])
        header, img = mx.recordio.unpack(sample)
        img = mx.image.imdecode(img)
        label = header.label
        self.current_index += 1
        return img, label

    def next(self):
        if self.current_batch < self.number_of_batches:
            self.current_batch += 1
            batch_data = mx.nd.zeros((self.batch_size, *self.output_size))
            labels = [
                mx.nd.zeros((self.batch_size)) for _ in range(self.number_of_labels)
            ]
            for i in range(self.batch_size):
                data, label = self.next_sample()
                if self.label_preprocess_func is not None:
                    label = self.label_preprocess_func(label)
                if self.image_preprocess_func is not None:
                    data = self.image_preprocess_func(data)
                batch_data[i] = mx.nd.transpose(data, axes=(2, 0, 1))
                for x in range(self.number_of_labels):
                    labels[x][i] = label[x]
            return mx.io.DataBatch([batch_data], labels)
        else:
            raise StopIteration

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
