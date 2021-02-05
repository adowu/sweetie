# -*- encoding: utf-8 -*-
'''
@Email :  wushaojun0107@gmail.com
@Time  :  02-04 2021
'''
import tensorflow as tf
from typing import Dict, Optional, Text, Sequence, List


class Model(tf.keras.Model):
    """
    call() 方法在模型初始化的时候，会调用一次，此时inputs 无论 generator yeild 的包含不包含 label，都不会有label

    而

    fit() evaluate() predict() ，这三个的inputs，在yeild是什么就是什么，所以 这三个的 都按照下标来取了；

    本可以不实现 call() 方法，但是发现 直接 model.save() 的时候，模型获取到输出的类型为NoneType 
    为了可以 model.save() 和 model.save_weights() 都能使用，就现在这样了。

    save() 方式由于不需要再定义一次模型结构，所以输出格式就是call()
    save_weights() 还是会根据 predict_step()定义的方式输出

    需要定义自己的Task 来完成 loss 计算
    还有一些坑后面再补吧！

    """

    def train_step(self, inputs) -> Dict:
        with tf.GradientTape() as tape:
            predictions = self.call(inputs[0], training=True)
            loss = self.task(
                labels=inputs[1], predictions=predictions, sample_weight=None)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables))
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        return metrics

    def test_step(self, inputs) -> Dict:
        predictions = self.call(inputs[0], training=True)
        loss = self.task(
            labels=inputs[1], predictions=predictions, sample_weight=None)
        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        return metrics

    def predict_step(self, inputs) -> Dict:
        predictions = self.call(inputs[0], training=False)
        result = {"predictions": predictions}
        return result

    def get_outputs(self, inputs, training: bool = False) -> tf.Tensor:
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        outputs = self.get_outputs(inputs, training=training)
        return outputs


class Metrics(tf.keras.layers.Layer):
    """Metric 基础类，按照 keras 的 Metric 实现了主要的方法"""

    def __init__(
        self,
        metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None,
        name: Text = "Metric",
    ) -> None:
        super().__init__(name=name)
        if metrics is None:
            metrics = [
                tf.keras.metrics.AUC(),
            ]
        self._metrics = metrics
        self.built = True

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Operation:
        update_ops = []
        for metric in self._metrics:
            update_ops.append(metric.update_state(
                y_true=y_true, y_pred=y_pred))

        return tf.group(update_ops)

    def reset_states(self) -> None:
        for metric in self.metrics:
            metric.reset_states()

    def result(self) -> List[tf.Tensor]:
        return [metric.result() for metric in self.metrics]


class Task(tf.keras.layers.Layer):
    """
    自定义loss
    """

    def __init__(
        self,
        loss: Optional[tf.keras.losses.Loss] = None,
        metrics: Optional[Metrics] = None,
        name: Optional[Text] = 'Task',
    ) -> None:
        super().__init__(name=name)
        self._loss = loss if loss is not None else tf.keras.losses.BinaryCrossentropy()
        self._customize_metrics = metrics

    @property
    def customize_metrics(self) -> Optional[Metrics]:
        return self._customize_metrics

    @customize_metrics.setter
    def customize_metrics(self, value: Optional[Metrics]) -> None:
        self._customize_metrics = value

    def call(
        self,
        labels: tf.Tensor,
        predictions: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        loss = self._loss(
            y_true=labels, y_pred=predictions, sample_weight=sample_weight
        )
        if not self._customize_metrics:
            return loss

        update_op = self._customize_metrics.update_state(labels, predictions)

        with tf.control_dependencies([update_op]):
            return tf.identity(loss)
