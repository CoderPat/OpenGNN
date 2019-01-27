"""Main library entrypoint."""

import random
from typing import Optional

import numpy as np
import tensorflow as tf
import json

from opengnn.models.model import Model


class Runner:
    """Class for managing training, inference, and export. It is mostly a
    wrapper around ``tf.estimator.Estimator``.
    """

    def __init__(self,
                 model: Model,
                 config,
                 seed: Optional[int]=None,
                 gpu_allow_growth: bool=False,
                 session_config: Optional[tf.ConfigProto]=None):
        """Initializes the runner parameters.

        Args:
            model: A :class:`opegnn.models.model.Model` instance to run.
            config: The run configuration.
            seed: The random seed to set.
            gpu_allow_growth: Allow GPU memory to grow dynamically.
            session_config: ``tf.ConfigProto`` overrides.
        """
        self.model = model
        self.config = config

        session_config_base = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(
                allow_growth=gpu_allow_growth))
        if session_config is not None:
            session_config_base.MergeFrom(session_config)
        session_config = session_config_base
        run_config = tf.estimator.RunConfig(
            model_dir=self.config["model_dir"],
            session_config=session_config,
            tf_random_seed=seed)

        if "train" in self.config:
            if "save_summary_steps" in self.config["train"]:
                run_config = run_config.replace(
                    save_summary_steps=self.config["train"]["save_summary_steps"],
                    log_step_count_steps=self.config["train"]["save_summary_steps"])
            if "save_checkpoints_steps" in self.config["train"]:
                run_config = run_config.replace(
                    save_checkpoints_secs=None,
                    save_checkpoints_steps=self.config["train"]["save_checkpoints_steps"])
            if "keep_checkpoint_max" in self.config["train"]:
                run_config = run_config.replace(
                    keep_checkpoint_max=self.config["train"]["keep_checkpoint_max"])

        tf.Session(config=session_config)

        np.random.seed(seed)
        random.seed(seed)

        self._estimator = tf.estimator.Estimator(
            self.model.model_fn(),
            config=run_config,
            params=self.config["params"],
            model_dir=self.config["model_dir"])

    def _build_train_spec(self) -> tf.estimator.TrainSpec:
        hooks = []
        early_stopping = self.config["eval"].get("early_stopping", None)
        
        if early_stopping is not None:
            if "patience" not in self.config["eval"]:
                raise ValueError("patience param must be set when using early stopping")
            try:
                early_stop_hook = tf.contrib.estimator.stop_if_no_increase_hook(
                    self._estimator,
                    early_stopping,
                    self.config["eval"]["patience"],
                    eval_dir=None,
                    min_steps=self.config["eval"]["patience"])
                hooks.append(early_stop_hook)
            except AttributeError:
                raise ValueError("Early Stopping requires Tensorflow 1.10 or above")
        
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.model.input_fn(
                tf.estimator.ModeKeys.TRAIN,
                self.config["train"]["batch_size"],
                self.config["data"],
                self.config["data"]["train_graphs_file"],
                labels_file=self.config["data"]["train_labels_file"],
                maximum_features_size=self.config["train"].get("maximum_features_size"),
                maximum_labels_size=self.config["train"].get("maximum_labels_size"),
                features_bucket_width=self.config["train"].get("features_bucket_width"),
                labels_bucket_width=self.config["train"].get("labels_bucket_width"),
                sample_buffer_size=self.config["train"].get("sample_buffer_size", 500000),
                num_threads=self.config["train"].get("num_threads")),
            max_steps=self.config["train"].get("train_steps"),
            hooks=hooks)
        return train_spec

    def _build_eval_spec(self) -> tf.estimator.EvalSpec:
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self.model.input_fn(
                tf.estimator.ModeKeys.EVAL,
                self.config["eval"].get("batch_size", 32),
                self.config["data"],
                self.config["data"]["eval_graphs_file"],
                maximum_features_size=self.config["eval"].get("maximum_features_size"),
                maximum_labels_size=self.config["eval"].get("maximum_labels_size"),
                features_bucket_width=self.config["eval"].get("features_bucket_width"),
                labels_bucket_width=self.config["eval"].get("labels_bucket_width"),
                labels_file=self.config["data"]["eval_labels_file"],
                num_threads=self.config["eval"].get("num_threads")),
            steps=None,
            throttle_secs=self.config["eval"].get("eval_delay", 3000))
        return eval_spec

    def train_and_evaluate(self) -> None:
        """Runs the training and evaluation loop."""
        if "eval" not in self.config:
            self.config["eval"] = {}

        in_memory = self.config["eval"].get("in_memory", False)

        train_spec = self._build_train_spec()
        eval_spec = self._build_eval_spec()

        if in_memory:
            try:
                evaluator = tf.contrib.estimator.InMemoryEvaluatorHook(
                    estimator=self._estimator,
                    input_fn=eval_spec.input_fn,
                    every_n_iter=self.config["eval"].get("eval_delay", 1000))
            except AttributeError:
                raise ValueError("In-Memory Evaluation requires Tensorflow 1.10 or above")

            self._estimator.train(
                train_spec.input_fn,
                max_steps=train_spec.max_steps,
                hooks=train_spec.hooks + (evaluator,))
        else:   
            tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)

    def train(self) -> None:
        """Runs the training loop."""
        train_spec = self._build_train_spec()
        self._estimator.train(
            train_spec.input_fn, max_steps=train_spec.max_steps)

    def infer(self,
              features_file,
              predictions_file):
        """Runs inference.

        Args:
            features_file: The file(s) to infer from.
            predictions_file: If set, predictions are saved in this file.
            checkpoint_path: Path of a specific checkpoint to predict. If ``None``,
            the latest is used.
            log_time: If ``True``, several time metrics will be printed in the logs at
            the end of the inference loop.
        """
        if "infer" not in self.config:
            self.config["infer"] = {}

        batch_size = self.config["infer"].get("batch_size", 1)
        input_fn = self.model.input_fn(
            tf.estimator.ModeKeys.PREDICT,
            batch_size,
            self.config["data"],
            features_file)

        with open(predictions_file, 'w') as out_file:
            for prediction in self._estimator.predict(input_fn=input_fn):
                proc_pred = self.model.process_prediction(prediction)
                out_file.write(json.dumps(proc_pred) + "\n")
