import mxnet as mx
from tqdm import tqdm
import mlflow


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()

class ImageClassifierTrainer:
    def __init__(self, output_symbol, label_names, training_iterator, validation_iterator=None, context = mx.gpu(), optimizer= 'adam'):
        self.model = mx.mod.Module(output_symbol, context=context, label_names=label_names)
        self.model.bind(data_shapes=training_iterator.provide_data, label_shapes=training_iterator.provide_label)
        self.model.init_params()
        self.model.init_optimizer(optimizer=optimizer, optimizer_params={'learning_rate': 0.01})
        self.label_names = label_names
        self.acc_metrics = [mx.metric.Accuracy() for _ in self.label_names]
        self.training_iterator = training_iterator
        self.validation_iterator = validation_iterator

    def train(self, epochs=30):
        for epoch in range(epochs):
            reset_metrics(self.acc_metrics)
            self.training_iterator.reset()
            for batch_index, batch in enumerate(self.training_iterator):
                self.model.forward(batch, is_train=True)
                outputs = self.model.get_outputs()
                self.model.backward()
                self.model.update()
                output_string = ''
                
                for label_number in range(len(self.label_names)):
                    self.acc_metrics[label_number].update(batch.label[label_number], outputs[label_number])
                    output_string += f'\tOutput{label_number}_acc:{self.acc_metrics[label_number].get()[1]:.3f}'
                print(f'\rEpoch:{epoch}\tBatch:{batch_index}'+output_string, end='', flush=True)
                
            if self.validation_iterator is not None:
                self.validation_iterator.reset()
                reset_metrics(self.acc_metrics)
                for batch in self.validation_iterator:
                    self.model.forward(batch, is_train=False)
                    outputs = self.model.get_outputs()
                    output_string = ''
                    for label_number in range(len(self.label_names)):
                        self.acc_metrics[label_number].update(batch.label[label_number], outputs[label_number])
                for label_number in range(len(self.label_names)):
                    output_string += f'\tOutput{label_number}_acc:{self.acc_metrics[label_number].get()[1]:.3f}'
                print(f'\nEpoch:{epoch}\tBatch:{batch_index}'+output_string)