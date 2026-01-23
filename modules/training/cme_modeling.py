##############################################################################################################
# Description: training and testing (algos, nn structure, loss functions,
# using validation loss to determine epoch number for training).
# this module should be interchangeable with other modules (
##############################################################################################################
import subprocess
import time
# types for type hinting
from typing import Tuple, List, Optional, Dict, Any, Generator

import matplotlib.pyplot as plt
import numpy as np
# imports
import tensorflow as tf
import wandb
from keras.regularizers import l2
from numpy import ndarray
from tensorflow.keras import layers, callbacks, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    LayerNormalization,
    Add
)

# from modules.reweighting.ImportanceWeighting import exDenseReweightsD
from modules.shared.globals import TARGET_MIN_NORM_WEIGHT, ES_CB_MONITOR
from modules.training.normlayer import NormalizeLayer
from modules.training.phase_manager import TrainingPhaseManager, IsTraining, create_weight_tensor_fast
from modules.training.sam_keras import SAMModel
from modules.training.ts_modeling import stratified_4fold_split


def error(z1, z2, label1, label2):
    """
    Computes the error between the squared L2 norm distance of two input predicted z values and the squared distance
    between their labels.

    :param z1: The predicted z value tensor for the first input sample.
    :param z2: The predicted z value tensor for the second input sample.
    :param label1: The label tensor of the first input sample.
    :param label2: The label tensor of the second input sample.
    :return: The squared difference tensor between the zdist and ydist.
    """
    # Compute the squared L2 norm distance between z1 and z2
    z_dist = tf.reduce_sum(tf.square(z1 - z2), axis=-1)

    # Compute the squared distance between label1 and label2
    y_dist = tf.square(label1 - label2)

    # Compute the squared difference between z_dist and y_dist
    squared_difference = tf.square(z_dist - y_dist)

    return squared_difference


def pds_space_norm(
        y_train: np.ndarray,
        rho: float = 1.5,
        lower_threshold: float = -0.5,
        upper_threshold: float = 0.5,
        y_min: float = None,
        y_max: float = None,
        debug: bool = False
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize the input labels according to the equation:
    y' = (Dz_max / (rho * Dy_max)) * y

    Parameters:
    - y_train (np.ndarray): The original labels to normalize.
    - rho (float): A factor allowing additional room outside the dataset for the representation space. Default is 1.5.
    - lower_threshold (float): The lower threshold value for normalization. Default is -0.5.
    - upper_threshold (float): The upper threshold value for normalization. Default is 0.5.
    - y_min (float): The minimum value in the original labels. Default is None. best when dealing with subset of larger dataset
    - y_max (float): The maximum value in the original labels. Default is None. best when dealing with subset of larger dataset
    - debug (bool): If True, show a sample of 5 instances before and after normalization. Default is False.

    Returns:
    - np.ndarray: The normalized labels.
    - float: The normalized lower threshold.
    - float: The normalized upper threshold.
    """
    # Define the maximum distance in the Z space
    Dz_max = 2

    # Calculate the maximum difference in the y labels
    if y_min is not None and y_max is not None:
        Dy_max = y_max - y_min  # Use the provided y_min and y_max
    else:
        Dy_max = np.max(y_train) - np.min(y_train)

    # Print the calculated Dy_max
    print(f"Dy_max: {Dy_max}")

    # Normalize the y labels
    y_normalized = (Dz_max / (rho * Dy_max)) * y_train

    # Normalize the threshold values
    norm_lower = (Dz_max / (rho * Dy_max)) * lower_threshold
    norm_upper = (Dz_max / (rho * Dy_max)) * upper_threshold

    # Debugging: Show a sample of 5 instances before and after normalization
    if debug:
        print("Sample of 5 instances before normalization:", y_train[:5])
        print("Sample of 5 instances after normalization:", y_normalized[:5])
        print(f"Normalized lower threshold: {norm_lower}")
        print(f"Normalized upper threshold: {norm_upper}")

    return y_normalized, norm_lower, norm_upper


def query_gpu_memory_usage():
    print("GPU MEMORY USAGE")
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'],
        stdout=subprocess.PIPE
    )
    output = result.stdout.decode('utf-8')
    print(output)


class ModelBuilder:
    """
    Class for building a neural network model.
    """

    # class variables
    debug = False

    def __init__(self, debug: bool = True) -> None:
        """
        Initialize the class variables.

        :param debug: Boolean to enable debug output.
        """
        self.debug = debug
        self.sep_sep_count = tf.Variable(0, dtype=tf.int32)
        self.sep_elevated_count = tf.Variable(0, dtype=tf.int32)
        self.sep_background_count = tf.Variable(0, dtype=tf.int32)
        self.elevated_elevated_count = tf.Variable(0, dtype=tf.int32)
        self.elevated_background_count = tf.Variable(0, dtype=tf.int32)
        self.background_background_count = tf.Variable(0, dtype=tf.int32)
        self.number_of_batches = 0

    def create_model(self,
                     input_dim: int,
                     embed_dim: int,
                     output_dim: int,
                     hiddens: List[int],
                     with_ae: bool = False) -> Model:
        """
        Create a neural network model with options for multiple heads using the Keras functional API.

        :param input_dim: Integer representing the number of input features.
        :param embed_dim: Integer representing the dimensionality of the feature (representation layer).
        :param output_dim: Integer representing the dimensionality of the output.
        :param hiddens: List of integers representing the number of nodes in each hidden layer.
        :param with_ae: Boolean flag to indicate whether to include an AutoEncoder (AE) head for input reconstruction.

        :return: The uncompiled model.
        """
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))
        x = input_layer

        # Define hidden layers according to architecture
        for nodes in hiddens:
            x = layers.Dense(nodes)(x)
            x = layers.LeakyReLU()(x)

        # Define the representation layer (Z features)
        repr_layer = layers.Dense(embed_dim)(x)
        repr_layer = layers.LeakyReLU(name='repr_layer')(repr_layer)

        # Add a regression head
        regression_head = layers.Dense(output_dim, activation='linear', name='regression_head')(repr_layer)

        # Create output_dim list
        outputs_list = [repr_layer, regression_head]

        # Add a decoder (AE) head for input reconstruction if with_ae is True
        if with_ae:
            decoder_head = repr_layer
            for nodes in reversed(hiddens):
                decoder_head = layers.Dense(nodes)(decoder_head)
                decoder_head = layers.LeakyReLU()(decoder_head)
            decoder_head = layers.Dense(input_dim, activation='linear', name='decoder_head')(decoder_head)
            outputs_list.append(decoder_head)

        # Create the model, repr, reg, decoder
        model = Model(inputs=input_layer, outputs=outputs_list)

        return model

    def create_model_pds(self,
                         input_dim: int,
                         embed_dim: int,
                         hiddens: List[int],
                         output_dim: Optional[int] = 1,
                         with_reg: bool = False, with_ae: bool = False) -> Model:
        """
        Create a neural network model with optional autoencoder and regression heads.
        The base model is used for feature extraction.

        :param input_dim: Integer representing the number of input features.
        :param embed_dim: Integer representing the dimensionality of the feature (representation layer).
        :param hiddens: List of integers representing the number of nodes in each hidden layer of the encoder.
        :param output_dim: Integer representing the dimensionality of the regression output. Default is 1.
        :param with_reg: Boolean flag to add a regression head to the model. Default is False.
        :param with_ae: Boolean flag to add a decoder to the model (making it an autoencoder). Default is False.
        :return: The uncompiled model with optional heads based on flags.
        """
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input
        for nodes in hiddens:
            x = layers.Dense(nodes)(x)
            x = layers.LeakyReLU()(x)

        x = layers.Dense(embed_dim)(x)
        x = layers.LeakyReLU()(x)
        repr_layer = NormalizeLayer(name='normalize_layer')(x)

        outputs = [repr_layer]

        # Optional Regression Head
        if with_reg:
            x_reg = repr_layer
            regression_output = layers.Dense(output_dim, activation='linear', name='regression_head')(x_reg)
            outputs.append(regression_output)

        # Optional Decoder
        if with_ae:
            x_dec = repr_layer
            for nodes in reversed(hiddens):
                x_dec = layers.Dense(nodes)(x_dec)
                x_dec = layers.LeakyReLU()(x_dec)
            decoder_output = layers.Dense(input_dim, activation='linear', name='decoder_head')(x_dec)
            outputs.append(decoder_output)

        # Complete model, repr, reg, decoder
        model = Model(inputs=encoder_input, outputs=outputs if len(outputs) > 1 else outputs[0])

        return model

    # def add_proj_head(self,
    #                   model: Model,
    #                   output_dim: int = 1,
    #                   hiddens: Optional[List[int]] = None,
    #                   freeze_features: bool = True,
    #                   pretraining: bool = False,
    #                   dropout: float = 0.0,
    #                   activation=None,
    #                   norm: str = None,
    #                   skipped_layers: int = 2,
    #                   name: str = 'mlp',
    #                   sam_rho: float = 0.05) -> Model:
    #     """
    #     Add a regression head with one output unit and a projection layer to an existing model,
    #     replacing the existing prediction layer and optionally the decoder layer.

    #     :param model: The existing model
    #     :param output_dim: The dimensionality of the output of the regression head.
    #     :param freeze_features: Whether to freeze the layers of the base model or not.
    #     :param hiddens: List of integers representing the hidden layers for the projection.
    #     :param pretraining: Whether to adapt the model for pretraining representations.
    #     :param dropout: Dropout rate for adding dropout layers.
    #     :param activation: Activation function to use. If None, defaults to LeakyReLU.
    #     :param norm: Type of normalization ('batch_norm' or 'layer_norm').
    #     :param skipped_layers: Number of layers between residual connections.
    #     :param name: Name of the model.
    #     :param sam_rho: Rho value for sharpness-aware minimization (SAM). Default is 0.05. if 0.0, SAM is not used.
    #     :return: The modified model with a projection layer and a regression head.
    #     """

    #     if hiddens is None:
    #         hiddens = [6]

    #     if activation is None:
    #         activation = LeakyReLU()

    #     residual = True if skipped_layers > 0 else False

    #     print(f'Features are frozen: {freeze_features}')

    #     # Determine the layer to be kept based on whether pretraining representations are used
    #     layer_to_keep = 'normalize_layer' if pretraining else 'repr_layer'

    #     # Remove the last layer(s) to keep only the representation layer
    #     new_base_model = Model(inputs=model.input, outputs=model.get_layer(layer_to_keep).output)

    #     # If freeze_features is True, freeze the layers of the new base model
    #     if freeze_features:
    #         for layer in new_base_model.layers:
    #             layer.trainable = False

    #     # Count existing dropout layers to avoid naming conflicts
    #     dropout_count = sum(1 for layer in model.layers if isinstance(layer, Dropout))

    #     # Extract the output of the last layer of the new base model (representation layer)
    #     repr_output = new_base_model.output

    #     # Projection Layer(s)
    #     x_proj = repr_output
    #     residual_layer = None

    #     for i, nodes in enumerate(hiddens):
    #         if i % skipped_layers == 0 and i > 0 and residual:
    #             if residual_layer is not None:
    #                 # Check if projection is needed
    #                 if x_proj.shape[-1] != residual_layer.shape[-1]:
    #                     # Correct projection to match 'x_proj' dimensions
    #                     residual_layer = Dense(x_proj.shape[-1], use_bias=False)(residual_layer)
    #                 x_proj = Add()([x_proj, residual_layer])
    #             residual_layer = x_proj  # Update the starting point for the next residual connection
    #         else:
    #             if i % skipped_layers == 0 or residual_layer is None:
    #                 residual_layer = x_proj

    #         x_proj = Dense(
    #             nodes,
    #             name=f"projection_layer_{i + 1}")(x_proj)

    #         if norm == 'batch_norm':
    #             x_proj = BatchNormalization(name=f"batch_norm_{i + 1}")(x_proj)
    #         elif norm == 'layer_norm':
    #             x_proj = LayerNormalization(name=f"layer_norm_{i + 1}")(x_proj)

    #         if callable(activation):
    #             x_proj = activation(x_proj)
    #         else:
    #             x_proj = LeakyReLU(name=f"activation_{i + 1}")(x_proj)

    #         if dropout > 0.0:
    #             x_proj = Dropout(dropout, name=f"proj_dropout_{dropout_count + i + 1}")(x_proj)

    #     # Add a Dense layer with one output unit for regression
    #     output_layer = Dense(output_dim, activation='linear', name=f"forecast_head")(x_proj)

    #     if sam_rho > 0.0:
    #         # create the new extended SAM model
    #         extended_model = SAMModel(inputs=new_base_model.input, outputs=[repr_output, output_layer], rho=sam_rho,
    #                                   name=name)
    #     else:
    #         # Create the new extended model
    #         extended_model = Model(inputs=new_base_model.input, outputs=[repr_output, output_layer], name=name)

    #     # If freeze_features is False, make all layers trainable
    #     if not freeze_features:
    #         for layer in extended_model.layers:
    #             layer.trainable = True

    #     return extended_model

    def overtrain_pds(self,
                      model: Model,
                      X_train: ndarray,
                      y_train: ndarray,
                      learning_rate: float = 1e-3,
                      epochs: int = 100,
                      batch_size: int = 32,
                      save_tag=None,
                      callbacks_list=None,
                      verbose: int = 1):
        """
            Trains the model and returns the training history.

            :param X_train: training and validation sets together
            :param y_train: labels of training and validation sets together
            :param save_tag: tag to use for saving experiments
            :param model: The TensorFlow model to stage2.
            :param learning_rate: The learning rate for the Adam optimizer.
            :param epochs: The maximum number of epochs for training.
            :param batch_size: The batch size for training.
            :param callbacks_list: List of callback instances to apply during training.
            :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


            :return: The training history as a History object.
            """

        # Compile the model
        model.compile(
            optimizer=AdamW(learning_rate=learning_rate),
            loss=self.pds_loss_vec
        )

        # model.compile(optimizer=AdamW(learning_rate=learning_rate), loss=self.pds_loss_vec)
        model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=callbacks_list,
                  verbose=verbose)

        # save the model weights
        model.save_weights(f"overfit_final_model_weights_{str(save_tag)}.h5")
        # print where the model weights are saved
        print(f"Model weights are saved in final_model_weights_{str(save_tag)}.h5")

    def overtrain_pds_inj_distr(self,
                                model: tf.keras.Model,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                learning_rate: float = 1e-3,
                                epochs: int = 100,
                                batch_size: int = 32,
                                lower_bound: float = -0.5,
                                upper_bound: float = 0.5,
                                save_tag=None,
                                callbacks_list=None,
                                strategy=None,
                                verbose: int = 1):
        """
        Trains the model and returns the training history with specific batch constraints in a distributed manner.

        :param strategy:  The distribution strategy to use for training.
        :param X_train: training and validation sets together
        :param y_train: labels of training and validation sets together
        :param save_tag: tag to use for saving experiments
        :param model: The TensorFlow model to train.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training, per replica.
        :param lower_bound: The lower bound for selecting rare samples.
        :param upper_bound: The upper bound for selecting rare samples.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        :return: The training history as a History object.
        """
        num_replicas = strategy.num_replicas_in_sync
        global_batch_size = batch_size * num_replicas

        # Identify injected rare samples
        rare_indices = np.where((y_train < lower_bound) | (y_train > upper_bound))[0]
        freq_indices = np.where((y_train >= lower_bound) & (y_train <= upper_bound))[0]

        if global_batch_size < len(rare_indices):
            raise ValueError(f"Global batch size must be at least the size of the injected rare samples. "
                             f"Current global batch size: {global_batch_size}, size of injected rare samples: {len(rare_indices)}")

        def data_generator(X, y, batch_size, rare_indices, freq_indices):
            while True:
                np.random.shuffle(freq_indices)
                for start in range(0, len(freq_indices), batch_size - len(rare_indices)):
                    end = min(start + batch_size - len(rare_indices), len(freq_indices))
                    freq_batch_indices = freq_indices[start:end]
                    batch_indices = np.concatenate([rare_indices, freq_batch_indices])
                    np.random.shuffle(batch_indices)
                    batch_X = X[batch_indices]
                    batch_y = y[batch_indices]
                    batch_y = batch_y.reshape(-1)
                    yield batch_X, batch_y

        with strategy.scope():
            dataset = tf.data.Dataset.from_generator(
                lambda: data_generator(X_train, y_train, global_batch_size, rare_indices, freq_indices),
                output_signature=(
                    tf.TensorSpec(shape=(None, X_train.shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32)
                )
            ).prefetch(tf.data.AUTOTUNE)

            dataset = strategy.experimental_distribute_dataset(dataset)

            def wrapped_loss(y_true, z_pred):
                # Determine the replica context and ID
                replica_context = tf.distribute.get_replica_context()
                replica_id = replica_context.replica_id_in_sync_group

                # Use replica_id directly as the quadrant index
                quadrant = tf.cast(replica_id, tf.int32)

                # Compute the loss for the assigned quadrant
                local_loss = self.pds_loss_vec_distr(y_true, z_pred, quadrant)

                # Aggregate losses from all replicas (workers)
                total_loss = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, local_loss)

                return total_loss / 4  # Average over the quadrants

            model.compile(
                optimizer=AdamW(learning_rate=learning_rate),
                loss=wrapped_loss
            )

            steps_per_epoch = len(freq_indices) // (global_batch_size - len(rare_indices))

            history = model.fit(
                dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=verbose
            )

            model.save_weights(f"overfit_final_model_weights_{str(save_tag)}.h5")
            print(f"Model weights are saved in overfit_final_model_weights_{str(save_tag)}.h5")

        return history

    # def process_batch_weights(self, batch_indices: np.ndarray, label_weights_dict: Dict[float, float]) -> np.ndarray:
    #     """
    #     Process a batch of indices to return the corresponding joint weights.
    # 
    #     :param batch_indices: A batch of sample indices.
    #     :param label_weights_dict: Dictionary containing label weights.
    #     :return: An array containing joint weights corresponding to the batch of indices.
    #     """
    #     # Convert list of tuples into a dictionary for O(1) lookup
    #     weight_dict = {pair: weight for pair, weight in zip(joint_weight_indices, joint_weights)}
    # 
    #     batch_weights = []
    #     for i in batch_indices:
    #         for j in batch_indices:
    #             if i < j:  # Only consider pairs (i, j) where i < j
    #                 weight = label_weights_dict
    #                 if weight is not None:
    #                     batch_weights.append(weight)
    # 
    #     return np.array(batch_weights)

    # def train_for_one_epoch(self,
    #                         model: tf.keras.Model,
    #                         optimizer: tf.keras.optimizers.Optimizer,
    #                         loss_fn,
    #                         X: np.ndarray,
    #                         y: np.ndarray,
    #                         batch_size: int,
    #                         label_weights_dict: Optional[Dict[float, float]] = None,
    #                         training: bool = True) -> float:
    #     """
    #     Train or evaluate the model for one epoch.
    #     processing the batches with indices is what making it slow
    #     :param model: The model to stage2 or evaluate.
    #     :param optimizer: The optimizer to use.
    #     :param loss_fn: The loss function to use.
    #     :param X: The feature set.
    #     :param y: The labels.
    #     :param batch_size: The batch size for training or evaluation.
    #     :param label_weights_dict: Dictionary containing label weights.
    #     :param training: Whether to apply training (True) or run evaluation (False).
    #     :return: The average loss for the epoch.
    #     """
    #     epoch_loss = 0.0
    #     num_batches = 0
    #
    #     for batch_idx in range(0, len(X), batch_size):
    #         batch_X = X[batch_idx:batch_idx + batch_size]
    #         batch_y = y[batch_idx:batch_idx + batch_size]
    #
    #         if len(batch_y) <= 1:
    #             # can't form a pair so skip
    #             continue
    #
    #         # print(f"batch_weights: {batch_weights}")
    #         # print(f"batch_y: {batch_y}")
    #         # print(f"batch_X: {batch_X}")
    #         with tf.GradientTape() as tape:
    #             predictions = model(batch_X, training=training)
    #             loss = loss_fn(batch_y, predictions, sample_weights=label_weights_dict)
    #
    #         if training:
    #             gradients = tape.gradient(loss, model.trainable_variables)
    #             # print(f"Gradients: {gradients}")
    #             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         epoch_loss += loss.numpy()
    #         num_batches += 1
    #
    #         print(f"batch: {num_batches}/{len(X) // batch_size}")
    #
    #     return epoch_loss / num_batches

    # def train_for_one_epoch_mh(
    #         self,
    #         model: tf.keras.Model,
    #         optimizer: tf.keras.optimizers.Optimizer,
    #         primary_loss_fn,
    #         X: np.ndarray,
    #         y: np.ndarray,
    #         batch_size: int,
    #         gamma_coeff: Optional[float] = None,
    #         lambda_coeff: Optional[float] = None,
    #         sample_weights: Optional[np.ndarray] = None,
    #         joint_weights: Optional[np.ndarray] = None,
    #         joint_weight_indices: Optional[List[Tuple[int, int]]] = None,
    #         with_reg=False,
    #         with_ae=False,
    #         training: bool = True) -> float:
    #     """
    #     Train the model for one epoch.
    #     processing the batches with indices is what making it slow
    #     :param with_ae:
    #     :param with_reg:
    #     :param model: The model to stage2.
    #     :param optimizer: The optimizer to use.
    #     :param primary_loss_fn: The primary loss function to use.
    #     :param X: The feature set.
    #     :param y: The labels.
    #     :param batch_size: The batch size for training.
    #     :param gamma_coeff: Coefficient for the regressor loss.
    #     :param lambda_coeff: Coefficient for the decoder loss.
    #     :param sample_weights: Individual sample weights.
    #     :param joint_weights: Optional array containing all joint weights for the dataset.
    #     :param joint_weight_indices: Optional list of tuples, each containing a pair of indices for which a joint weight exists.
    #     :param training: Whether to apply training or evaluation (default is True for training).
    #     :return: The average loss for the epoch.
    #     """
    #
    #     epoch_loss = 0.0
    #     num_batches = 0
    #
    #     for batch_idx in range(0, len(X), batch_size):
    #         batch_X = X[batch_idx:batch_idx + batch_size]
    #         batch_y = y[batch_idx:batch_idx + batch_size]
    #         batch_sample_weights = None if sample_weights is None \
    #             else sample_weights[batch_idx:batch_idx + batch_size]
    #
    #         if len(batch_y) <= 1:
    #             # can't form a pair so skip
    #             continue
    #
    #         # Get the corresponding joint weights for this batch
    #         batch_weights = None
    #         if joint_weights is not None and joint_weight_indices is not None:
    #             batch_weights = self.process_batch_weights(
    #                 np.arange(batch_idx, batch_idx + batch_size), joint_weights, joint_weight_indices)
    #
    #         with tf.GradientTape() as tape:
    #             outputs = model(batch_X, training=training)
    #
    #             # Unpack the outputs based on the model configuration
    #             if with_reg and with_ae:
    #                 primary_predictions, regressor_predictions, decoder_predictions = outputs
    #             elif with_reg:
    #                 primary_predictions, regressor_predictions = outputs
    #                 decoder_predictions = None
    #             elif with_ae:
    #                 primary_predictions, decoder_predictions = outputs
    #                 regressor_predictions = None
    #             else:
    #                 primary_predictions = outputs
    #                 regressor_predictions, decoder_predictions = None, None
    #
    #             # Primary loss
    #             primary_loss = primary_loss_fn(batch_y, primary_predictions, sample_weights=batch_weights)
    #
    #             # Regressor loss
    #             regressor_loss = 0
    #             if with_reg and gamma_coeff is not None:
    #                 regressor_loss = tf.keras.losses.mean_squared_error(batch_y, regressor_predictions)
    #                 if batch_sample_weights is not None:
    #                     regressor_loss = tf.cast(regressor_loss, batch_sample_weights.dtype)
    #                     regressor_loss = tf.reduce_sum(regressor_loss * batch_sample_weights) / tf.reduce_sum(
    #                         batch_sample_weights)
    #                 regressor_loss *= gamma_coeff
    #
    #             # Decoder loss
    #             decoder_loss = 0
    #             if with_ae and lambda_coeff is not None:
    #                 decoder_loss = tf.keras.losses.mean_squared_error(batch_X, decoder_predictions)
    #                 decoder_loss *= lambda_coeff
    #
    #             # Make sure all loss tensors have the same dtype
    #             dtype_to_use = tf.float32  # or tf.float64 based on your preference
    #
    #             primary_loss = tf.cast(primary_loss, dtype_to_use)
    #             regressor_loss = tf.cast(regressor_loss, dtype_to_use)
    #             decoder_loss = tf.cast(decoder_loss, dtype_to_use)
    #
    #             # Total loss
    #             total_loss = primary_loss + regressor_loss + decoder_loss
    #
    #         if training:
    #             gradients = tape.gradient(total_loss, model.trainable_variables)
    #             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #         # Make sure total_loss is reduced to a single scalar value.
    #         total_loss_scalar = tf.reduce_sum(total_loss)
    #
    #         # Update epoch_loss
    #         epoch_loss += total_loss_scalar.numpy()
    #
    #         num_batches += 1
    #
    #         print(f"batch: {num_batches}/{len(X) // batch_size}")
    #
    #     return epoch_loss / num_batches

    def train_pds(self,
                  model: tf.keras.Model,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  X_subtrain: np.ndarray,
                  y_subtrain: np.ndarray,
                  X_val: np.ndarray,
                  y_val: np.ndarray,
                  train_label_weights_dict: Optional[Dict[float, float]] = None,
                  val_label_weights_dict: Optional[Dict[float, float]] = None,
                  learning_rate: float = 1e-3,
                  epochs: int = 100,
                  batch_size: int = 32,
                  patience: int = 9,
                  weight_decay: float = 0.0,
                  momentum_beta1: float = 0.9,
                  save_tag: Optional[str] = None,
                  callbacks_list=None,
                  verbose: int = 1) -> dict:
        """
        Custom training loop to stage2 the model and returns the training history.

        :param X_train: training and validation sets together
        :param y_train: labels of training and validation sets together
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param model: The TensorFlow model to train.
        :param train_label_weights_dict: Dictionary containing label weights for the training set.
        :param val_label_weights_dict: Dictionary containing label weights for the validation set.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :param weight_decay: The L2 regularization factor.
        :param momentum_beta1: The beta1 parameter for the Adam optimizer.
        :param save_tag: Tag to use for saving experiments.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


        :return: The training history as a dictionary.
        """

        pm = TrainingPhaseManager()

        if callbacks_list is None:
            callbacks_list = []

        # Add the IsTraining callback to the list
        callbacks_list.append(IsTraining(pm))

        # Initialize early stopping and model checkpointing for subtraining
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=False
        )
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"model_weights_{str(save_tag)}.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )

        # Append the early stopping and checkpoint callbacks to the custom callbacks list
        subtrain_callbacks_list = callbacks_list + [early_stopping_cb, checkpoint_cb]

        # Save initial weights for retraining on full training set after best epoch found
        initial_weights = model.get_weights()

        # Optimizer and history initialization
        model.compile(
            optimizer=AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta_1=momentum_beta1
            ),
            loss=lambda y_true, y_pred: self.pds_loss_vec(
                y_true, y_pred,
                phase_manager=pm,
                train_sample_weights=train_label_weights_dict,
                val_sample_weights=val_label_weights_dict,
            )
        )

        history = model.fit(
            X_subtrain, y_subtrain,
            epochs=epochs,
            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
            validation_data=(X_val, y_val),
            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
            callbacks=subtrain_callbacks_list,
            verbose=verbose
        )

        # Determine the optimal number of epochs from the fit history
        best_epoch = np.argmin(history.history['val_loss']) + 1  # +1 to adjust for 0-based index

        # Retraining on the combined dataset
        print(f"Retraining to the best epoch: {best_epoch}")

        # Reset model weights to initial state before retraining
        model.set_weights(initial_weights)

        model.compile(
            optimizer=AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                beta_1=momentum_beta1
            ),
            loss=lambda y_true, y_pred: self.pds_loss_vec(
                y_true, y_pred,
                phase_manager=pm,
                train_sample_weights=train_label_weights_dict,
            )
        )

        model.fit(
            X_train, y_train,
            epochs=best_epoch,
            batch_size=batch_size if batch_size > 0 else len(y_train),
            callbacks=callbacks_list,
            verbose=verbose
        )

        # Save the final model
        model.save_weights(f"final_model_weights_{str(save_tag)}.h5")
        # print where the model weights are saved
        print(f"Model weights are saved in final_model_weights_{str(save_tag)}.h5")

        return history

    # def train_pds_folds(
    #         self,
    #         model: tf.keras.Model,
    #         X_train: np.ndarray,
    #         y_train: np.ndarray,
    #         train_label_weights_dict: Optional[Dict[float, float]] = None,
    #         alpha: float = 0.5,
    #         alphaV: float = 1,
    #         seed: int = 42,
    #         bandwidth: float = 0.1,
    #         learning_rate: float = 1e-3,
    #         epochs: int = 100,
    #         batch_size: int = 32,
    #         patience: int = 9,
    #         weight_decay: float = 0.0,
    #         momentum_beta1: float = 0.9,
    #         save_tag: Optional[str] = None,
    #         callbacks_list=None,
    #         verbose: int = 1
    # ) -> None:
    #     """
    #     Custom training loop to stage2 the model and returns the training history.

    #     :param model: The TensorFlow model to train.
    #     :param X_train: training and validation sets together
    #     :param y_train: labels of training and validation sets together
    #     :param alpha: The alpha value for the PDS validation loss
    #     :param alphaV: The alpha value for the PDS validation loss
    #     :param seed: The seed for the random number generator.
    #     :param bandwidth: The bandwidth for the PDS label reweighting.
    #     :param train_label_weights_dict: Dictionary containing label weights for the training set.
    #     :param learning_rate: The learning rate for the Adam optimizer.
    #     :param epochs: The maximum number of epochs for training.
    #     :param batch_size: The batch size for training.
    #     :param patience: The number of epochs with no improvement to wait before early stopping.
    #     :param weight_decay: The L2 regularization factor.
    #     :param momentum_beta1: The beta1 parameter for the Adam optimizer.
    #     :param save_tag: Tag to use for saving experiments.
    #     :param callbacks_list: List of callback instances to apply during training.
    #     :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    #     """

    #     pm = TrainingPhaseManager()

    #     if callbacks_list is None:
    #         callbacks_list = []

    #     # Add the IsTraining callback to the list
    #     callbacks_list.append(IsTraining(pm))

    #     # Initialize early stopping and model checkpointing for subtraining
    #     early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    #         monitor='val_loss',
    #         patience=patience,
    #         restore_best_weights=False
    #     )
    #     checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    #         filepath=f"model_weights_{str(save_tag)}.h5",
    #         monitor='val_loss',
    #         save_best_only=True,
    #         save_weights_only=True
    #     )

    #     # Append the early stopping and checkpoint callbacks to the custom callbacks list
    #     subtrain_callbacks_list = callbacks_list + [early_stopping_cb, checkpoint_cb]

    #     # Save initial weights for retraining on full training set after best epoch found
    #     initial_weights = model.get_weights()

    #     # 4-fold cross-validation
    #     folds_optimal_epochs = []
    #     for fold_idx, (X_subtrain, y_subtrain, X_val, y_val) in enumerate(
    #             stratified_4fold_split(X_train, y_train, seed=seed, shuffle=True)):
    #         print(f'Fold: {fold_idx}')
    #         # print all cme_files shapes
    #         print(f'X_subtrain.shape: {X_subtrain.shape}, y_subtrain.shape: {y_subtrain.shape}')
    #         print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

    #         # Compute the sample weights for subtraining
    #         delta_subtrain = y_subtrain[:, 0]
    #         print(f'delta_subtrain.shape: {delta_subtrain.shape}')
    #         print(f'rebalancing the subtraining set...')
    #         min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_subtrain)
    #         subtrain_weights_dict = exDenseReweightsD(
    #             X_subtrain, delta_subtrain,
    #             alpha=alpha, bw=bandwidth,
    #             min_norm_weight=min_norm_weight,
    #             debug=False).label_reweight_dict
    #         print(f'subtraining set rebalanced.')

    #         # Compute the sample weights for validation
    #         delta_val = y_val[:, 0]
    #         print(f'delta_val.shape: {delta_val.shape}')
    #         print(f'rebalancing the validation set...')
    #         min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_val)
    #         val_weights_dict = exDenseReweightsD(
    #             X_val, delta_val,
    #             alpha=alphaV, bw=bandwidth,
    #             min_norm_weight=min_norm_weight,
    #             debug=False).label_reweight_dict
    #         print(f'validation set rebalanced.')

    #         # Optimizer and history initialization
    #         model.compile(
    #             optimizer=AdamW(
    #                 learning_rate=learning_rate,
    #                 weight_decay=weight_decay,
    #                 beta_1=momentum_beta1
    #             ),
    #             loss=lambda y_true, y_pred: self.pds_loss_vec(
    #                 y_true, y_pred,
    #                 phase_manager=pm,
    #                 train_sample_weights=subtrain_weights_dict,
    #                 val_sample_weights=val_weights_dict,
    #             )
    #         )

    #         history = model.fit(
    #             X_subtrain, y_subtrain,
    #             epochs=epochs,
    #             batch_size=batch_size if batch_size > 0 else len(y_subtrain),
    #             validation_data=(X_val, y_val),
    #             validation_batch_size=batch_size if batch_size > 0 else len(y_val),
    #             callbacks=subtrain_callbacks_list,
    #             verbose=verbose
    #         )

    #         # optimal epoch for fold
    #         folds_optimal_epochs.append(np.argmin(history.history[ES_CB_MONITOR]) + 1)
    #         # wandb log the fold's optimal
    #         print(f'fold_{fold_idx}_best_epoch: {folds_optimal_epochs[-1]}')
    #         wandb.log({f'fold_{fold_idx}_best_epoch': folds_optimal_epochs[-1]})

    #         # Reset model weights to initial state before retraining
    #         model.set_weights(initial_weights)

    #     # determine the optimal number of epochs from the folds
    #     optimal_epochs = int(np.mean(folds_optimal_epochs))
    #     print(f'optimal_epochs: {optimal_epochs}')
    #     wandb.log({'optimal_epochs': optimal_epochs})

    #     model.compile(
    #         optimizer=AdamW(
    #             learning_rate=learning_rate,
    #             weight_decay=weight_decay,
    #             beta_1=momentum_beta1
    #         ),
    #         loss=lambda y_true, y_pred: self.pds_loss_vec(
    #             y_true, y_pred,
    #             phase_manager=pm,
    #             train_sample_weights=train_label_weights_dict,
    #             val_sample_weights=None
    #         )
    #     )

    #     model.fit(
    #         X_train, y_train,
    #         epochs=optimal_epochs,
    #         batch_size=batch_size if batch_size > 0 else len(y_train),
    #         callbacks=callbacks_list,
    #         verbose=verbose
    #     )

    #     # Save the final model
    #     model.save_weights(f"final_model_weights_{str(save_tag)}.h5")
    #     # print where the model weights are saved
    #     print(f"Model weights are saved in final_model_weights_{str(save_tag)}.h5")

    def overtrain_pds_dl(self,
                         model: tf.keras.Model,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         train_label_weights_dict: Optional[Dict[float, float]] = None,
                         learning_rate: float = 1e-3,
                         epochs: int = 100,
                         batch_size: int = 32,
                         save_tag: Optional[str] = None,
                         callbacks_list=None,
                         verbose: int = 1) -> Dict[str, List[Any]]:
        """
        Custom training loop to stage2 the model and returns the training history.

        :param X_train: training and validation sets together
        :param y_train: labels of training and validation sets together
        :param model: The TensorFlow model to stage2.
        :param train_label_weights_dict: Dictionary containing label weights for the stage2 set.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param save_tag: Tag to use for saving experiments.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.


        :return: The training history as a dictionary.
        """

        if callbacks_list is None:
            callbacks_list = []

        # Setting up callback environment
        params = {
            'epochs': epochs,
            'steps': None,
            'verbose': verbose,
            'do_validation': False,
            'metrics': ['loss'],
        }
        for cb in callbacks_list:
            cb.set_model(model)
            cb.set_params(params)

        logs = {}
        # Signal the beginning of training
        for cb in callbacks_list:
            cb.on_train_begin(logs=logs)

        # Optimizer and history initialization
        optimizer = AdamW(learning_rate=learning_rate)
        model.compile(optimizer=optimizer)  # Set the optimizer for the model

        # Retraining on the combined dataset
        # Reset history for retraining
        retrain_history = {'loss': []}

        # Retrain up to the best epoch
        for epoch in range(epochs):
            for cb in callbacks_list:
                cb.on_epoch_begin(epoch, logs=logs)

            retrain_loss = self.train_for_one_epoch(
                model, optimizer,
                self.pds_loss_vec,
                X_train, y_train,
                batch_size=batch_size if batch_size > 0 else len(y_train),
                label_weights_dict=train_label_weights_dict)

            # Log the retrain loss
            retrain_history['loss'].append(retrain_loss)
            print(f"Retrain Epoch {epoch + 1}/{epochs}, Loss: {retrain_loss}")

            logs = {'loss': retrain_loss}  # Update logs with retrain loss

            for cb in callbacks_list:
                cb.on_epoch_end(epoch, logs=logs)

        for cb in callbacks_list:
            cb.on_train_end(logs=logs)

        # Save the final model
        model.save_weights(f"overfit_final_model_weights_{str(save_tag)}.h5")
        # print where the model weights are saved
        print(f"Model weights are saved in final_model_weights_{str(save_tag)}.h5")

        return retrain_history

    def overtrain_pds_inj(self,
                          model: tf.keras.Model,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          train_label_weights_dict: Optional[Dict[float, float]] = None,
                          learning_rate: float = 1e-3,
                          epochs: int = 100,
                          batch_size: int = 32,
                          rare_injection_count: int = -1,
                          lower_bound: float = -0.5,
                          upper_bound: float = 0.5,
                          save_tag: Optional[str] = None,
                          callbacks_list=None,
                          verbose: int = 1) -> Dict[str, List[Any]]:
        """
        Custom training loop to train the model with sample weights and injected rare samples.

        :param X_train: training and validation sets together
        :param y_train: labels of training and validation sets together
        :param model: The TensorFlow model to train.
        :param train_label_weights_dict: Dictionary containing label weights for the training set.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param rare_injection_count: Number of rare samples to inject in each batch (-1 for all, 0 for none, default 2).
        :param lower_bound: The lower bound for selecting rare samples.
        :param upper_bound: The upper bound for selecting rare samples.
        :param save_tag: Tag to use for saving experiments.
        :param callbacks_list: List of callback instances to apply during training.
        :param verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        :return: The training history as a dictionary.
        """

        rare_indices = np.where((y_train < lower_bound) | (y_train > upper_bound))[0]
        freq_indices = np.where((y_train >= lower_bound) & (y_train <= upper_bound))[0]

        if rare_injection_count == -1:
            rare_injection_count = len(rare_indices)
        elif rare_injection_count > len(rare_indices):
            rare_injection_count = len(rare_indices)
            print(f"rare_injection_count ({rare_injection_count}) is greater than the number of rare samples "
                  f"({len(rare_indices)}).")
        else:
            # initial number of batches = number of samples / batch size
            num_batches = len(y_train) // batch_size
            ratio = len(rare_indices) / num_batches
            if ratio > rare_injection_count:
                # insert ratio  / rare_injection_count rare samples in each batch
                rare_injection_count = int(ratio / rare_injection_count)
                print(f"Adjusting rare_injection_count to {ratio} based on the ratio of rare samples to batches.")
            else:
                # insert rare_injection_count rare samples in each batch
                # rare_injection_count = rare_injection_count
                print(f"Injecting {rare_injection_count} rare samples in each batch.")

        steps_per_epoch = len(freq_indices) // (batch_size - rare_injection_count)

        # Check if the batch size is sufficient
        if batch_size < rare_injection_count:
            raise ValueError(f"Batch size must be at least the number of injected rare samples. "
                             f"Current batch size: {batch_size}, rare_injection_count: {rare_injection_count}")

        # def data_generator(X, y, batch_size, rare_indices, freq_indices, rare_injection_count):
        #     while True:
        #         np.random.shuffle(freq_indices)
        #         for start in range(0, len(freq_indices), batch_size - rare_injection_count):
        #             end = min(start + batch_size - rare_injection_count, len(freq_indices))
        #             freq_batch_indices = freq_indices[start:end]
        #             rare_sample_indices = np.random.choice(rare_indices, rare_injection_count, replace=False)
        #             batch_indices = np.concatenate([rare_sample_indices, freq_batch_indices])
        #             np.random.shuffle(batch_indices)
        #             yield X[batch_indices], y[batch_indices]
        def data_generator(X, y, batch_size, rare_indices, freq_indices, rare_injection_count):
            # Initialize the start index for rare samples
            rare_start_index = 0

            while True:
                np.random.shuffle(freq_indices)  # Shuffle frequent indices for each epoch
                rare_indices = np.random.permutation(
                    rare_indices)  # Shuffle rare indices only at the start of each epoch

                for start in range(0, len(freq_indices), batch_size - rare_injection_count):
                    end = min(start + batch_size - rare_injection_count, len(freq_indices))
                    freq_batch_indices = freq_indices[start:end]

                    # Calculate the end index for rare samples in this batch
                    rare_end_index = rare_start_index + rare_injection_count
                    if rare_end_index > len(rare_indices):
                        # If we exceed the list, wrap around
                        rare_sample_indices = np.concatenate(
                            [rare_indices[rare_start_index:], rare_indices[:rare_end_index - len(rare_indices)]]
                        )
                        rare_start_index = rare_end_index - len(rare_indices)  # Update start index for next batch
                    else:
                        # Select consecutive rare samples
                        rare_sample_indices = rare_indices[rare_start_index:rare_end_index]
                        rare_start_index = rare_end_index  # Update start index for next batch

                    # Reset rare index if needed
                    if rare_start_index >= len(rare_indices):
                        rare_start_index = 0

                    batch_indices = np.concatenate([rare_sample_indices, freq_batch_indices])
                    np.random.shuffle(batch_indices)  # Shuffle indices to mix rare and frequent samples
                    yield X[batch_indices], y[batch_indices]

        model.compile(
            optimizer=AdamW(learning_rate=learning_rate),
            loss=lambda y_true, y_pred: self.pds_loss_vec(
                y_true, y_pred, sample_weights=train_label_weights_dict
            )
        )

        history = model.fit(
            data_generator(X_train, y_train, batch_size, rare_indices, freq_indices, rare_injection_count),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=verbose
        )

        model.save_weights(f"overfit_final_model_weights_{str(save_tag)}.h5")
        print(f"Model weights are saved in overfit_final_model_weights_{str(save_tag)}.h5")

        return history

    def create_tf_dataset(
            self,
            X: np.ndarray,
            y: np.ndarray,
            rare_indices: np.ndarray,
            freq_indices: np.ndarray,
            batch_size: int,
            rare_injection_count: int
    ) -> tf.data.Dataset:
        """
        Creates a TensorFlow dataset from the data generator.

        :param X: Feature set.
        :param y: Labels.
        :param rare_indices: Indices of the rare samples.
        :param freq_indices: Indices of the frequent samples.
        :param batch_size: Size of each batch.
        :param rare_injection_count: Number of rare samples to inject in each batch.

        :return: A tf.data.Dataset object.
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(
                X, y,
                rare_indices,
                freq_indices,
                batch_size,
                rare_injection_count
            ),
            output_signature=(
                tf.TensorSpec(shape=(None, X.shape[1]), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
            )
        )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def data_generator(
            self,
            X: np.ndarray,
            y: np.ndarray,
            rare_indices: np.ndarray,
            freq_indices: np.ndarray,
            batch_size: int,
            rare_injection_count: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generalized data generator to yield batches with a mixture of rare and frequent samples.

        :param X: Feature set.
        :param y: Labels.
        :param rare_indices: Indices of the rare samples.
        :param freq_indices: Indices of the frequent samples.
        :param batch_size: Size of each batch.
        :param rare_injection_count: Number of rare samples to inject in each batch.

        :yield: Batches of (features, labels) with mixed rare and frequent samples.
        """
        # Initialize the start index for rare samples
        rare_start_index = 0

        while True:
            np.random.shuffle(freq_indices)  # Shuffle frequent indices for each epoch
            rare_indices = np.random.permutation(rare_indices)  # Shuffle rare indices only at the start of each epoch

            for start in range(0, len(freq_indices), batch_size - rare_injection_count):
                end = min(start + batch_size - rare_injection_count, len(freq_indices))
                freq_batch_indices = freq_indices[start:end]

                # Calculate the end index for rare samples in this batch
                rare_end_index = rare_start_index + rare_injection_count
                if rare_end_index > len(rare_indices):
                    # If we exceed the list, wrap around
                    rare_sample_indices = np.concatenate(
                        [rare_indices[rare_start_index:], rare_indices[:rare_end_index - len(rare_indices)]]
                    )
                    rare_start_index = rare_end_index - len(rare_indices)  # Update start index for next batch
                else:
                    # Select consecutive rare samples
                    rare_sample_indices = rare_indices[rare_start_index:rare_end_index]
                    rare_start_index = rare_end_index  # Update start index for next batch

                # Reset rare index if needed
                if rare_start_index >= len(rare_indices):
                    rare_start_index = 0

                batch_indices = np.concatenate([rare_sample_indices, freq_batch_indices])
                np.random.shuffle(batch_indices)  # Shuffle indices to mix rare and frequent samples
                # Extract the actual data (features and labels) for the current batch
                batch_X = X[batch_indices]
                batch_y = y[batch_indices].reshape(-1)
                # Yield the current batch (features and labels) to be used by the training loop
                yield batch_X, batch_y

    def calc_steps_per_epoch(
            self,
            freq_indices: np.ndarray,
            batch_size: int,
            rare_injection_count: int
    ) -> int:
        """
        Calculates the number of steps per epoch.

        :param freq_indices: Indices of frequent samples.
        :param batch_size: Batch size.
        :param rare_injection_count: Number of rare samples injected per batch.
        :return: Number of steps per epoch.
        """
        return len(freq_indices) // (batch_size - rare_injection_count)

    def id_rares_freqs(
            self,
            y: np.ndarray,
            lower_bound: float,
            upper_bound: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identifies indices of rare and frequent samples based on specified bounds.

        :param y: Labels.
        :param lower_bound: Lower bound for frequent samples.
        :param upper_bound: Upper bound for frequent samples.
        :return: Tuple of (rare_indices, freq_indices).
        """
        rare_indices = np.where((y < lower_bound) | (y > upper_bound))[0]
        freq_indices = np.where((y >= lower_bound) & (y <= upper_bound))[0]
        return rare_indices, freq_indices

    def adjust_rare_inj_count(
            self,
            rare_injection_count: int,
            rare_indices: np.ndarray,
            freq_indices: np.ndarray,
            batch_size: int,
            phase_name: str
    ) -> int:
        """
        Adjusts the rare injection count based on the number of rare samples and batch size.

        :param rare_injection_count: Initial rare injection count.
        :param rare_indices: Indices of rare samples.
        :param freq_indices: Indices of frequent samples.
        :param batch_size: Batch size.
        :param phase_name: Name of the training phase ('subtrain' or 'train').
        :return: Adjusted rare injection count.
        """
        adjusted_count = rare_injection_count
        num_rare = len(rare_indices)
        num_batches = len(freq_indices) // (batch_size - rare_injection_count)

        if adjusted_count == -1:
            adjusted_count = num_rare
        elif adjusted_count > num_rare:
            adjusted_count = num_rare
            print(f"rare_injection_count_{phase_name} ({rare_injection_count}) is greater than "
                  f"the number of rare samples ({num_rare}).")
        else:
            ratio = num_rare / num_batches
            if ratio > adjusted_count:
                adjusted_count = int(ratio / adjusted_count)
                print(f"Adjusting rare_injection_count_{phase_name} to {adjusted_count} "
                      f"based on the ratio of rare samples to batches.")
            else:
                print(f"Injecting {adjusted_count} rare samples in each {phase_name} batch.")

        if batch_size < adjusted_count:
            raise ValueError(f"Batch size must be at least the number of injected rare samples for {phase_name}. "
                             f"Current batch size: {batch_size}, rare_injection_count_{phase_name}: {adjusted_count}")

        return adjusted_count

    def inject_batch_dataset(
            self,
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            rare_injection_count: int,
            lower_bound: float,
            upper_bound: float,
            phase_name: str
    ) -> Tuple[tf.data.Dataset, int]:
        """
        Creates a dataset with injected rare samples and calculates steps per epoch.

        :param X: Feature set.
        :param y: Labels.
        :param batch_size: Batch size.
        :param rare_injection_count: Number of rare samples to inject in each batch.
        :param lower_bound: Lower bound for frequent samples.
        :param upper_bound: Upper bound for frequent samples.
        :param phase_name: Name of the training phase ('subtrain' or 'train').

        :return: Tuple of (dataset, steps_per_epoch).
        """
        # Identify rare and frequent indices
        rare_indices, freq_indices = self.id_rares_freqs(y, lower_bound, upper_bound)

        # Adjust rare injection count
        adjusted_rare_injection_count = self.adjust_rare_inj_count(
            rare_injection_count,
            rare_indices,
            freq_indices,
            batch_size,
            phase_name
        )

        # Calculate steps per epoch
        steps_per_epoch = self.calc_steps_per_epoch(
            freq_indices,
            batch_size,
            adjusted_rare_injection_count
        )

        # Create dataset
        dataset = self.create_tf_dataset(
            X, y,
            rare_indices,
            freq_indices,
            batch_size,
            adjusted_rare_injection_count
        )

        return dataset, steps_per_epoch

    def train_pds_dl_heads(
            self,
            model: tf.keras.Model,
            X_subtrain: np.ndarray,
            y_subtrain: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_joint_weights: Optional[np.ndarray] = None,
            sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
            val_sample_joint_weights: Optional[np.ndarray] = None,
            val_sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
            train_sample_joint_weights: Optional[np.ndarray] = None,
            train_sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
            sample_weights: Optional[np.ndarray] = None,
            val_sample_weights: Optional[np.ndarray] = None,
            train_sample_weights: Optional[np.ndarray] = None,
            with_reg: bool = False,
            with_ae: bool = False,
            learning_rate: float = 1e-3,
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 9,
            save_tag: Optional[str] = None) -> dict:
        """
        Custom training loop to stage2 the model and returns the training history.

        :param y_train:
        :param X_train:
        :param train_sample_joint_weights:
        :param train_sample_joint_weights_indices:
        :param train_sample_weights:
        :param with_ae:
        :param with_reg:
        :param sample_weights:
        :param val_sample_weights:
        :param model: The TensorFlow model to stage2.
        :param X_subtrain: The training feature set.
        :param y_subtrain: The training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param sample_joint_weights: The reweighting factors for pairs of labels in training set.
        :param sample_joint_weights_indices: Indices of the reweighting factors in training set.
        :param val_sample_joint_weights: The reweighting factors for pairs of labels in validation set.
        :param val_sample_joint_weights_indices: Indices of the reweighting factors in validation set.
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The maximum number of epochs for training.
        :param batch_size: The batch size for training.
        :param patience: The number of epochs with no improvement to wait before early stopping.
        :param save_tag: Tag to use for saving experiments.
        :return: The training history as a dictionary.
        """

        # Initialize early stopping and best epoch variables
        best_val_loss = float('inf')
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_for_estimation = 5

        gamma_coeff, lambda_coeff = self.estimate_gamma_lambda_coeffs(
            model, X_subtrain, y_subtrain, self.pds_loss_vec,
            sample_weights, sample_joint_weights, sample_joint_weights_indices,
            learning_rate=learning_rate, n_epochs=epochs_for_estimation,
            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
            with_ae=with_ae, with_reg=with_reg)

        print(f'found gamma: {gamma_coeff}, lambda: {lambda_coeff}')

        # Initialize TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        #
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Optimizer and history initialization
        optimizer = AdamW(learning_rate=learning_rate)
        history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            train_loss = self.train_for_one_epoch_mh(
                model, optimizer, self.pds_loss_vec, X_subtrain, y_subtrain,
                batch_size=batch_size if batch_size > 0 else len(y_subtrain)
                , gamma_coeff=gamma_coeff, lambda_coeff=lambda_coeff,
                sample_weights=sample_weights, joint_weights=sample_joint_weights,
                joint_weight_indices=sample_joint_weights_indices, with_reg=with_reg, with_ae=with_ae)

            val_loss = self.train_for_one_epoch_mh(
                model, optimizer, self.pds_loss_vec, X_val, y_val,
                batch_size=batch_size if batch_size > 0 else len(y_val),
                gamma_coeff=gamma_coeff, lambda_coeff=lambda_coeff,
                sample_weights=val_sample_weights, joint_weights=val_sample_joint_weights,
                joint_weight_indices=val_sample_joint_weights_indices, with_reg=with_reg, with_ae=with_ae,
                training=False)

            # Log and save epoch losses
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss}, Validation Loss: {val_loss}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save the model weights
                model.save_weights(f"best_model_weights_{str(save_tag)}.h5")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping triggered.")
                    break

        # Plotting the losses
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig(f"training_plot_{str(save_tag)}.png")
        plt.close()

        # Retraining on the combined dataset
        print(f"Retraining to the best epoch: {best_epoch}")

        # Reset history for retraining
        retrain_history = {'loss': []}

        # Retrain up to the best epoch
        for epoch in range(best_epoch):
            retrain_loss = self.train_for_one_epoch_mh(
                model, optimizer, self.pds_loss_vec, X_train, y_train,
                batch_size=batch_size if batch_size > 0 else len(y_train),
                gamma_coeff=gamma_coeff, lambda_coeff=lambda_coeff,
                sample_weights=train_sample_weights,
                joint_weights=train_sample_joint_weights,
                joint_weight_indices=train_sample_joint_weights_indices,
                with_reg=with_reg, with_ae=with_ae)

            # Log the retrain loss
            retrain_history['loss'].append(retrain_loss)
            print(f"Retrain Epoch {epoch + 1}/{best_epoch}, Loss: {retrain_loss}")

        # Save the final model
        model.save_weights(f"final_model_weights_{str(save_tag)}.h5")

        return history

    def train_reg_head(self,
                       model: Model,
                       X_subtrain: ndarray,
                       y_subtrain: ndarray,
                       X_val: ndarray,
                       y_val: ndarray,
                       X_train: ndarray,
                       y_train: ndarray,
                       sample_weights: Optional[ndarray] = None,
                       sample_val_weights: Optional[ndarray] = None,
                       sample_train_weights: Optional[ndarray] = None,
                       learning_rate: float = 1e-3,
                       epochs: int = 100,
                       batch_size: int = 32,
                       patience: int = 9,
                       save_tag=None) -> callbacks.History:
        """
        Train a neural network model focusing only on the regression output.
        Include reweighting for balancing the loss.

        :param save_tag:
        :param model: The neural network model.
        :param X_subtrain: sub Training features.
        :param y_subtrain: sub Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param sample_weights: Sample weights for sub training set.
        :param sample_val_weights: Sample weights for validation set.
        :param sample_train_weights: Sample weights for training
        :param learning_rate: Learning rate for Adam optimizer.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Number of epochs for early stopping.
        :return: Training history.
        """

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_regression_head_loss', patience=patience,
                                                    restore_best_weights=True)
        # Setup model checkpointing
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_{str(save_tag)}.h5", save_weights_only=True)
        # Compile the model
        model.compile(optimizer=AdamW(learning_rate=learning_rate), loss={'regression_head': 'mse'})

        # Train the model with a validation set
        history = model.fit(X_subtrain, {'regression_head': y_subtrain},
                            sample_weight=sample_weights,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=(X_val, {'regression_head': y_val}, sample_val_weights),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
                            callbacks=[early_stopping_cb, checkpoint_cb])

        # Find the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_regression_head_loss']) + 1

        # Plot training and validation loss
        plt.plot(history.history['regression_head_loss'], label='Training Loss')
        plt.plot(history.history['val_regression_head_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        file_path = f"training_reg_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Retrain the model to the best epoch using combined cme_files
        model.fit(X_train, {'regression_head': y_train},
                  sample_weight=sample_train_weights,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb])

        # save the model weights
        model.save_weights(f"extended_model_weights_{str(save_tag)}.h5")

        return history

    def estimate_gamma_lambda_coeffs(self,
                                     model: tf.keras.Model,
                                     X_subtrain: np.ndarray,
                                     y_subtrain: np.ndarray,
                                     primary_loss_fn,
                                     sample_weights: Optional[np.ndarray] = None,
                                     sample_joint_weights: Optional[np.ndarray] = None,
                                     sample_joint_weights_indices: Optional[List[Tuple[int, int]]] = None,
                                     learning_rate: float = 1e-3, n_epochs: int = 10,
                                     batch_size: int = 32,
                                     with_ae=False, with_reg=False) -> Tuple[float, float]:
        """
        Estimate the gamma and lambda coefficients for balancing the primary, regression, and decoder losses.

        :param with_ae:
        :param with_reg:
        :param sample_joint_weights:
        :param sample_joint_weights_indices:
        :param model: The neural network model.
        :param X_subtrain: Training features.
        :param y_subtrain: Training labels.
        :param primary_loss_fn: Primary loss function.
        :param sample_weights: Sample weights for training set.
        :param sample_val_weights: Sample weights for validation set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param n_epochs: Number of epochs to stage2 each branch for coefficient estimation.
        :param batch_size: Batch size.
        :return: Estimated gamma and lambda coefficients.
        """

        # Initialize lists to store validation losses for each head
        primary_losses = []
        optimizer = AdamW(learning_rate=learning_rate)
        # Train the primary head using custom training loop
        for epoch in range(n_epochs):
            train_loss = self.train_for_one_epoch_mh(
                model, optimizer, primary_loss_fn,
                X_subtrain, y_subtrain,
                batch_size,
                sample_weights=sample_weights,
                joint_weights=sample_joint_weights,
                joint_weight_indices=sample_joint_weights_indices,
                training=True,
                with_ae=with_ae, with_reg=with_reg)
            primary_losses.append(train_loss)

        reg_losses = []
        dec_losses = []

        # Train regression branch only if with_reg is True
        if with_reg:
            model.compile(optimizer=AdamW(learning_rate=learning_rate),
                          loss={'regression_head': 'mse'})
            history_reg = model.fit(X_subtrain, {'regression_head': y_subtrain},
                                    sample_weight=sample_weights,
                                    epochs=n_epochs,
                                    batch_size=batch_size)
            reg_losses = history_reg.history['loss']

        # Train decoder branch only if with_ae is True
        if with_ae:
            model.compile(optimizer=AdamW(learning_rate=learning_rate),
                          loss={'decoder_head': 'mse'})
            history_dec = model.fit(X_subtrain, {'decoder_head': X_subtrain},
                                    sample_weight=sample_weights,
                                    epochs=n_epochs,
                                    batch_size=batch_size)
            dec_losses = history_dec.history['loss']

        # # Initialize coefficients to None
        # gamma_coef = None
        # lambda_coef = None
        #
        # # Calculate gamma and lambda as the sum of the ratios, if applicable
        # if with_reg:
        #     gamma_ratios = [p / r for p, r in zip(primary_losses, reg_losses)]
        #     gamma_coef = np.mean(gamma_ratios)
        #
        # if with_ae:
        #     lambda_ratios = [p / d for p, d in zip(primary_losses, dec_losses)]
        #     lambda_coef = np.mean(lambda_ratios)

        # Initialize coefficients to None
        gamma_coef = None
        lambda_coef = None

        # Assuming primary_losses, reg_losses, and dec_losses are lists or could be converted to NumPy arrays
        primary_losses = np.array(primary_losses)

        # Calculate gamma and lambda as the mean of the ratios, if applicable
        if with_reg:
            reg_losses = np.array(reg_losses)
            gamma_ratios = primary_losses / reg_losses
            gamma_coef = np.mean(gamma_ratios)

        if with_ae:
            dec_losses = np.array(dec_losses)
            lambda_ratios = primary_losses / dec_losses
            lambda_coef = np.mean(lambda_ratios)

        return gamma_coef, lambda_coef

    def estimate_lambda_coef(self,
                             model,
                             X_subtrain, y_subtrain,
                             sample_weights=None,
                             learning_rate=1e-3,
                             n_epochs=10,
                             batch_size=32):
        """
        Estimate the lambda coefficient for balancing the regression and decoder losses.

        :param model: The neural network model.
        :param X_subtrain: Training features.
        :param y_subtrain: Training labels.
        :param sample_weights: Sample weights for training set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param n_epochs: Number of epochs to stage2 each branch for lambda estimation.
        :param batch_size: Batch size.
        :return: Estimated lambda coefficient.
        """

        # Train regression branch only
        model.compile(optimizer=AdamW(learning_rate=learning_rate),
                      loss={'regression_head': 'mse'})
        history_reg = model.fit(X_subtrain, {'regression_head': y_subtrain},
                                sample_weight=sample_weights,
                                epochs=n_epochs,
                                batch_size=batch_size)

        reg_losses = history_reg.history['loss']

        # Train decoder branch only
        model.compile(optimizer=AdamW(learning_rate=learning_rate),
                      loss={'decoder_head': 'mse'})
        history_dec = model.fit(X_subtrain, {'decoder_head': X_subtrain},
                                sample_weight=sample_weights,
                                epochs=n_epochs,
                                batch_size=batch_size)

        dec_losses = history_dec.history['loss']

        # Calculate lambda as the sum of the ratios
        # ratios = [r / d for r, d in zip(reg_losses, dec_losses)]
        # lambda_coef = np.mean(ratios)
        reg_losses = np.array(reg_losses)
        dec_losses = np.array(dec_losses)

        # Compute ratios using element-wise division
        ratios = reg_losses / dec_losses

        # Calculate the mean of the ratios
        lambda_coef = np.mean(ratios)

        return lambda_coef

    def train_reg_ae_heads(self, model: Model,
                           X_subtrain: ndarray,
                           y_subtrain: ndarray,
                           X_val: ndarray,
                           y_val: ndarray,
                           X_train: ndarray,
                           y_train: ndarray,
                           sample_weights: Optional[ndarray] = None,
                           sample_val_weights: Optional[ndarray] = None,
                           sample_train_weights: Optional[ndarray] = None,
                           learning_rate: float = 1e-3,
                           epochs: int = 100,
                           batch_size: int = 32,
                           patience: int = 9,
                           save_tag=None) -> callbacks.History:
        """
        Train a neural network model focusing on the regression and autoencoder output.
        Includes reweighting for balancing the loss and saves the model weights.

        :param model: The neural network model.
        :param X_subtrain: Training features.
        :param y_subtrain: Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :param sample_weights: Sample weights for training set.
        :param sample_val_weights: Sample weights for validation set.
        :param learning_rate: Learning rate for Adam optimizer.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param patience: Number of epochs for early stopping.
        :param save_tag: Tag for saving model weights and plots.
        :return: Training history.
        """

        epochs_for_estimation = 5

        lambda_coef = self.estimate_lambda_coef(model, X_subtrain, y_subtrain,
                                                sample_weights,
                                                learning_rate, epochs_for_estimation,
                                                batch_size=batch_size if batch_size > 0 else len(y_subtrain))

        print(f"Lambda coefficient found: {lambda_coef}")

        # Setup TensorBoard
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # print("Run the command line:\n tensorboard --logdir logs/fit")

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Model checkpointing
        checkpoint_cb = callbacks.ModelCheckpoint(f"model_weights_ae_{str(save_tag)}.h5", save_weights_only=True)

        # Compile the model
        model.compile(optimizer=AdamW(learning_rate=learning_rate),
                      loss={'regression_head': 'mse', 'decoder_head': 'mse'},
                      loss_weights={'regression_head': 1.0, 'decoder_head': lambda_coef})

        # Prepare cme_files dictionary
        y_dict = {'regression_head': y_subtrain, 'decoder_head': X_subtrain}
        val_y_dict = {'regression_head': y_val, 'decoder_head': X_val}

        # Train the model
        history = model.fit(X_subtrain, y_dict,
                            sample_weight=sample_weights,
                            epochs=epochs,
                            batch_size=batch_size if batch_size > 0 else len(y_subtrain),
                            validation_data=(X_val, val_y_dict, sample_val_weights),
                            validation_batch_size=batch_size if batch_size > 0 else len(y_val),
                            callbacks=[early_stopping_cb, checkpoint_cb])

        # Find the best epoch from early stopping
        best_epoch = np.argmin(history.history['val_loss']) + 1

        # Plot training and validation loss
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        file_path = f"training_ae_plot_{str(save_tag)}.png"
        plt.savefig(file_path)
        plt.close()

        # Retrain the model to the best epoch using combined cme_files
        model.fit(X_train, {'regression_head': y_train, 'decoder_head': X_train},
                  sample_weight=sample_train_weights,
                  epochs=best_epoch,
                  batch_size=batch_size if batch_size > 0 else len(y_train),
                  callbacks=[checkpoint_cb])

        # Save the extended model weights
        model.save_weights(f"extended_model_weights_ae_{str(save_tag)}.h5")

        return history

    def plot_model(self, model: Model, name: str) -> None:
        """
        Plot the model architecture and save the figure.
        :param name: name of the file
        :param model: The model to plot.
        :return: None
        """
        tf.keras.utils.plot_model(model, to_file=f'./{name}.png', show_shapes=True, show_layer_names=True)

    @tf.autograph.experimental.do_not_convert
    def pdc_loss_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the PDC (Pairwise Distance Correlation) loss with diagonal terms excluded,
        using absolute differences for labels and L2 norm for representations.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1]
            z_pred: A batch of predicted feature vectors, shape of [batch_size, n_features]
            phase_manager: Manager that tracks training/validation phase
            train_sample_weights: Dictionary mapping label values to weights during training
            val_sample_weights: Dictionary mapping label values to weights during validation
            reduction: Type of reduction to apply (Note: loss is scalar, reduction has no effect)

        Returns:
            The PDC loss value as a scalar tensor
        """
        # Cast tensors to float32 for stability
        batch_size = tf.shape(y_true)[0]
        dtype = tf.float32  # Use high precision for stability
        y_true = tf.cast(y_true, dtype)
        z_pred = tf.cast(z_pred, dtype)
        epsilon = tf.keras.backend.epsilon()

        # Compute distance matrices
        # For labels: absolute difference instead of squared difference
        y_diff = tf.abs(y_true - tf.transpose(y_true))

        # For representations: L2 norm instead of squared difference
        z_diff = tf.sqrt(tf.reduce_sum(
            tf.square(z_pred[:, tf.newaxis, :] - z_pred[tf.newaxis, :, :]),
            axis=-1
        ) + epsilon)

        y_diff = tf.cast(y_diff, dtype)

        off_diag_size = tf.cast(batch_size * (batch_size - 1), dtype)

        # Compute means excluding diagonal terms
        Dy_mean = tf.reduce_sum(y_diff) / off_diag_size
        Dz_mean = tf.reduce_sum(z_diff) / off_diag_size

        # Center the variables
        Dy_centered = y_diff - Dy_mean
        Dz_centered = z_diff - Dz_mean

        # Create weights matrix
        weights_matrix = tf.ones((batch_size, batch_size), dtype=dtype)

        # Apply sample weights if provided
        sample_weights = train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        if sample_weights is not None:
            weights = tf.squeeze(create_weight_tensor_fast(y_true, sample_weights), axis=-1)
            weights_matrix = tf.cast(weights[:, None] * weights[None, :], dtype)

        # Reshape diagonal zeros to match weights_matrix shape
        diag_zeros = tf.zeros([batch_size], dtype=dtype)
        weights_matrix = tf.linalg.set_diag(weights_matrix, diag_zeros)

        # Compute moments
        cov_Dy_Dz = tf.reduce_sum(weights_matrix * Dy_centered * Dz_centered)
        var_Dy = tf.reduce_sum(weights_matrix * tf.square(Dy_centered))
        var_Dz = tf.reduce_sum(weights_matrix * tf.square(Dz_centered))

        # Compute correlation
        pcc = cov_Dy_Dz / tf.sqrt((var_Dy * var_Dz) + epsilon)

        return 1.0 - pcc


    @tf.autograph.experimental.do_not_convert
    def pdc_loss_vec_cos(
        self,
        y_true: tf.Tensor,
        z_pred: tf.Tensor,
        phase_manager: TrainingPhaseManager,
        train_sample_weights: Optional[Dict[float, float]] = None,
        val_sample_weights: Optional[Dict[float, float]] = None,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the PDC (Pairwise Distance Correlation) loss with diagonal terms excluded,
        using absolute differences for labels and cosine distance (1 - dot product) for unit norm representations.
        
        Args:
            y_true: A batch of true label values, shape of [batch_size, 1]
            z_pred: A batch of predicted feature vectors (assumed unit norm), shape of [batch_size, n_features]
            phase_manager: Manager that tracks training/validation phase
            train_sample_weights: Dictionary mapping label values to weights during training
            val_sample_weights: Dictionary mapping label values to weights during validation
            reduction: Type of reduction to apply (Note: loss is scalar, reduction has no effect)
        
        Returns:
            The PDC loss value using cosine distance as a scalar tensor
        """
        # Cast tensors to float32 for stability
        batch_size = tf.shape(y_true)[0]
        dtype = tf.float32
        y_true = tf.cast(y_true, dtype)
        z_pred = tf.cast(z_pred, dtype)
        epsilon = tf.keras.backend.epsilon()
        
        # Compute distance matrices
        # For labels: absolute difference (same as original)
        y_diff = tf.abs(y_true - tf.transpose(y_true))
        
        # For representations: cosine distance (1 - dot product) for unit norm vectors
        # Compute pairwise dot products: z_pred[i] · z_pred[j]
        dot_products = tf.matmul(z_pred, tf.transpose(z_pred))
        
        # Cosine distance = 1 - dot_product (for unit norm vectors)
        # Clamp dot products to [-1, 1] to handle numerical errors
        dot_products = tf.clip_by_value(dot_products, -1.0, 1.0)
        z_diff = 1.0 - dot_products
        
        # Ensure z_diff is non-negative (cosine distance should be in [0, 2])
        z_diff = tf.maximum(z_diff, 0.0)
        
        y_diff = tf.cast(y_diff, dtype)
        
        off_diag_size = tf.cast(batch_size * (batch_size - 1), dtype)
        
        # Compute means excluding diagonal terms
        Dy_mean = tf.reduce_sum(y_diff) / off_diag_size
        Dz_mean = tf.reduce_sum(z_diff) / off_diag_size
        
        # Center the variables
        Dy_centered = y_diff - Dy_mean
        Dz_centered = z_diff - Dz_mean
        
        # Create weights matrix
        weights_matrix = tf.ones((batch_size, batch_size), dtype=dtype)
        
        # Apply sample weights if provided
        sample_weights = train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        if sample_weights is not None:
            weights = tf.squeeze(create_weight_tensor_fast(y_true, sample_weights), axis=-1)
            weights_matrix = tf.cast(weights[:, None] * weights[None, :], dtype)
        
        # Reshape diagonal zeros to match weights_matrix shape
        diag_zeros = tf.zeros([batch_size], dtype=dtype)
        weights_matrix = tf.linalg.set_diag(weights_matrix, diag_zeros)
        
        # Compute moments
        cov_Dy_Dz = tf.reduce_sum(weights_matrix * Dy_centered * Dz_centered)
        var_Dy = tf.reduce_sum(weights_matrix * tf.square(Dy_centered))
        var_Dz = tf.reduce_sum(weights_matrix * tf.square(Dz_centered))
        
        # Compute correlation
        pcc = cov_Dy_Dz / tf.sqrt((var_Dy * var_Dz) + epsilon)
        
        return 1.0 - pcc

    @tf.autograph.experimental.do_not_convert
    def pdc_loss_vec_geo(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the PDC (Pairwise Distance Correlation) loss with diagonal terms excluded,
        using absolute differences for labels and geodesic distance on unit sphere for representations.
        Checks if z_pred is normalized and normalizes if necessary.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1]
            z_pred: A batch of predicted feature vectors, shape of [batch_size, n_features]
            phase_manager: Manager that tracks training/validation phase
            train_sample_weights: Dictionary mapping label values to weights during training
            val_sample_weights: Dictionary mapping label values to weights during validation
            reduction: Type of reduction to apply (Note: loss is scalar, reduction has no effect)

        Returns:
            The PDC loss value as a scalar tensor using geodesic distance
        """
        # Cast tensors to float32 for stability
        batch_size = tf.shape(y_true)[0]
        dtype = tf.float32  # Use high precision for stability
        y_true = tf.cast(y_true, dtype)
        z_pred = tf.cast(z_pred, dtype)
        epsilon = tf.keras.backend.epsilon()

        # Check if features are normalized and normalize if necessary
        norms = tf.norm(z_pred, axis=-1)
        is_normalized = tf.reduce_all(tf.abs(norms - 1.0) < 1e-6)
        z_pred_norm = tf.cond(
            is_normalized,
            lambda: z_pred,  # Already normalized
            lambda: tf.nn.l2_normalize(z_pred, axis=-1)  # Normalize
        )

        # Compute distance matrices
        # For labels: absolute difference (same as original)
        y_diff = tf.abs(y_true - tf.transpose(y_true))

        # For representations: geodesic distance on unit sphere
        # Compute pairwise dot products
        dot_products = tf.matmul(z_pred_norm, z_pred_norm, transpose_b=True)
        
        # Clamp dot products to [-1, 1] to handle numerical errors
        dot_products = tf.clip_by_value(dot_products, -1.0 + epsilon, 1.0 - epsilon)
        
        # Geodesic distance = arccos(dot_product) 
        z_diff = tf.acos(dot_products)

        y_diff = tf.cast(y_diff, dtype)

        off_diag_size = tf.cast(batch_size * (batch_size - 1), dtype)

        # Compute means excluding diagonal terms
        Dy_mean = tf.reduce_sum(y_diff) / off_diag_size
        Dz_mean = tf.reduce_sum(z_diff) / off_diag_size

        # Center the variables
        Dy_centered = y_diff - Dy_mean
        Dz_centered = z_diff - Dz_mean

        # Create weights matrix
        weights_matrix = tf.ones((batch_size, batch_size), dtype=dtype)

        # Apply sample weights if provided
        sample_weights = train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        if sample_weights is not None:
            weights = tf.squeeze(create_weight_tensor_fast(y_true, sample_weights), axis=-1)
            weights_matrix = tf.cast(weights[:, None] * weights[None, :], dtype)

        # Reshape diagonal zeros to match weights_matrix shape
        diag_zeros = tf.zeros([batch_size], dtype=dtype)
        weights_matrix = tf.linalg.set_diag(weights_matrix, diag_zeros)

        # Compute moments
        cov_Dy_Dz = tf.reduce_sum(weights_matrix * Dy_centered * Dz_centered)
        var_Dy = tf.reduce_sum(weights_matrix * tf.square(Dy_centered))
        var_Dz = tf.reduce_sum(weights_matrix * tf.square(Dz_centered))

        # Compute correlation
        pcc = cov_Dy_Dz / tf.sqrt((var_Dy * var_Dz) + epsilon)

        return 1.0 - pcc

    

    def pdc_loss_linear_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the PDC loss with diagonal terms excluded, using a single operation
        for computing all pairwise differences. (The default PDC for now)

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1]
            z_pred: A batch of predicted feature vectors, shape of [batch_size, n_features]
            phase_manager: Manager that tracks training/validation phase
            train_sample_weights: Dictionary mapping label values to weights during training
            val_sample_weights: Dictionary mapping label values to weights during validation
            reduction: Type of reduction to apply (Note: loss is scalar, reduction has no effect)

        Returns:
            The PDC loss value as a scalar tensor
        """
        batch_size = tf.shape(y_true)[0]
        dtype = z_pred.dtype
        y_true = tf.cast(y_true, dtype)
        eps = tf.keras.backend.epsilon()

        # Create indices for all pairs we want to compute
        indices1 = tf.concat([tf.range(batch_size - 1), tf.zeros(batch_size - 2, dtype=tf.int32)], axis=0)
        indices2 = tf.concat([tf.range(1, batch_size), tf.range(2, batch_size)], axis=0)

        # Compute all differences at once using gathered indices
        z_pairs_diff = tf.gather(z_pred, indices1) - tf.gather(z_pred, indices2)
        y_pairs_diff = tf.gather(y_true, indices1) - tf.gather(y_true, indices2)

        # Compute distances
        z_diff = tf.sqrt(tf.reduce_sum(tf.square(z_pairs_diff), axis=-1) + eps)
        y_diff = tf.abs(tf.squeeze(y_pairs_diff))

        # Rest of the computations remain the same
        # Compute means with epsilon for stability
        Dy_mean = tf.reduce_mean(y_diff)
        Dz_mean = tf.reduce_mean(z_diff)

        # Center the variables
        Dy_centered = y_diff - Dy_mean
        Dz_centered = z_diff - Dz_mean

        # Create weights
        weights = tf.ones((2 * batch_size - 3,), dtype=dtype)

        if sample_weights := (train_sample_weights if phase_manager.is_training_phase() else val_sample_weights):
            weights = tf.concat(
                [
                    create_weight_tensor_fast(tf.gather(y_true, indices1), sample_weights)
                    * create_weight_tensor_fast(tf.gather(y_true, indices2), sample_weights),
                ],
                axis=0,
            )
            weights = tf.cast(weights, dtype=dtype)

        # Compute moments with numerical stability
        cov_Dy_Dz = tf.reduce_sum(weights * Dy_centered * Dz_centered)
        var_Dy = tf.reduce_sum(weights * tf.square(Dy_centered + eps))
        var_Dz = tf.reduce_sum(weights * tf.square(Dz_centered + eps))

        # Compute correlation with enhanced stability
        denominator = tf.sqrt((var_Dy * var_Dz) + eps)

        pcc = cov_Dy_Dz / denominator

        return 1.0 - pcc

    def pdc_loss_linear_m_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the PDC loss with specific pairs:
        - 0 to batch_size - 2 paired with 1 to batch_size - 1
        - The first element paired with the last element.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1].
            z_pred: A batch of predicted feature vectors, shape of [batch_size, n_features].
            phase_manager: Manager that tracks training/validation phase.
            train_sample_weights: Dictionary mapping label values to weights during training.
            val_sample_weights: Dictionary mapping label values to weights during validation.
            reduction: Type of reduction to apply (Note: loss is scalar, reduction has no effect).

        Returns:
            The PDC loss value as a scalar tensor.
        """
        batch_size = tf.shape(y_true)[0]
        dtype = z_pred.dtype
        y_true = tf.cast(y_true, dtype)
        eps = tf.keras.backend.epsilon()

        # Create indices for the selected pairs
        indices1 = tf.range(batch_size - 1)  # 0 -> batch_size - 2
        indices2 = tf.range(1, batch_size)  # 1 -> batch_size - 1
        first_last_indices1 = tf.constant([0], dtype=tf.int32)  # First element
        first_last_indices2 = tf.expand_dims(batch_size - 1, axis=0)  # Last element (Tensor-compatible)

        # Concatenate indices for the two sets of pairs
        indices1 = tf.concat([indices1, first_last_indices1], axis=0)
        indices2 = tf.concat([indices2, first_last_indices2], axis=0)

        # Compute all differences at once using gathered indices
        z_pairs_diff = tf.gather(z_pred, indices1) - tf.gather(z_pred, indices2)
        y_pairs_diff = tf.gather(y_true, indices1) - tf.gather(y_true, indices2)

        # Compute distances
        z_diff = tf.sqrt(tf.reduce_sum(tf.square(z_pairs_diff), axis=-1) + eps)
        y_diff = tf.abs(tf.squeeze(y_pairs_diff))

        # Compute means with epsilon for stability
        Dy_mean = tf.reduce_mean(y_diff)
        Dz_mean = tf.reduce_mean(z_diff)

        # Center the variables
        Dy_centered = y_diff - Dy_mean
        Dz_centered = z_diff - Dz_mean

        # Create weights
        weights = tf.ones_like(y_diff, dtype=dtype)

        if sample_weights := (train_sample_weights if phase_manager.is_training_phase() else val_sample_weights):
            weights = tf.concat(
                [
                    create_weight_tensor_fast(tf.gather(y_true, indices1), sample_weights)
                    * create_weight_tensor_fast(tf.gather(y_true, indices2), sample_weights),
                ],
                axis=0,
            )
            weights = tf.cast(weights, dtype=dtype)

        # Compute moments with numerical stability
        cov_Dy_Dz = tf.reduce_sum(weights * Dy_centered * Dz_centered)
        var_Dy = tf.reduce_sum(weights * tf.square(Dy_centered + eps))
        var_Dz = tf.reduce_sum(weights * tf.square(Dz_centered + eps))

        # Compute correlation with enhanced stability
        denominator = tf.sqrt((var_Dy * var_Dz) + eps)

        pcc = cov_Dy_Dz / denominator

        return 1.0 - pcc

    def pds_loss_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the weighted loss for a batch of predicted features and their labels,
        using absolute differences for labels and L2 norm for representations.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1].
            z_pred: A batch of predicted Z values, shape of [batch_size, n_features].
            phase_manager: Manager that tracks training/validation phase.
            train_sample_weights: Dictionary mapping label values to weights during training.
            val_sample_weights: Dictionary mapping label values to weights during validation.
            reduction: Type of reduction to apply to the loss.

        Returns:
            The weighted average error for all unique combinations of the samples in the batch.
        """
        batch_size = tf.shape(y_true)[0]
        dtype = tf.float32  # Use high precision for stability
        y_true = tf.cast(y_true, dtype)
        z_pred = tf.cast(z_pred, dtype)
        epsilon = tf.keras.backend.epsilon()

        # Compute pairwise differences using broadcasting
        y_diff = tf.abs(y_true - tf.transpose(y_true))
        z_diff = tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    z_pred[:, tf.newaxis, :] - z_pred[tf.newaxis, :, :]
                ), axis=-1
            ) + epsilon
        )

        # Exclude diagonal elements
        num_comparisons = tf.cast(batch_size * (batch_size - 1), dtype)
        y_diff = tf.cast(y_diff, dtype)  # Ensure y_diff is in the same dtype as z_diff

        # Compute the pairwise loss
        pairwise_loss = tf.square(z_diff - y_diff)

        # Create weights matrix
        weights_matrix = tf.ones((batch_size, batch_size), dtype=dtype)

        # Apply sample weights if provided
        sample_weights = train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        if sample_weights is not None:
            weights = create_weight_tensor_fast(y_true, sample_weights)
            # Ensure weights is 1D by squeezing any extra dimensions
            weights = tf.squeeze(weights)
            weights_matrix = tf.cast(weights[:, None] * weights[None, :], dtype)

        # Reshape diagonal zeros to match weights_matrix shape
        diag_zeros = tf.zeros([batch_size], dtype=dtype)
        weights_matrix = tf.linalg.set_diag(weights_matrix, diag_zeros)

        # Apply the weights to the pairwise loss
        pairwise_loss *= weights_matrix

        # Calculate total weighted error
        total_error = tf.reduce_sum(pairwise_loss)

        # Apply reduction
        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error * 0.5  # Account for symmetry in pairs
        elif reduction == tf.keras.losses.Reduction.NONE:
            return total_error / num_comparisons
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def pds_loss_vec_cos(
        self,
        y_true: tf.Tensor,
        z_pred: tf.Tensor,
        phase_manager: TrainingPhaseManager,
        train_sample_weights: Optional[Dict[float, float]] = None,
        val_sample_weights: Optional[Dict[float, float]] = None,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the weighted loss for a batch of predicted features and their labels,
        using absolute differences for labels and cosine distance (1 - dot product) for unit norm representations.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1].
            z_pred: A batch of predicted Z values (assumed unit norm), shape of [batch_size, n_features].
            phase_manager: Manager that tracks training/validation phase.
            train_sample_weights: Dictionary mapping label values to weights during training.
            val_sample_weights: Dictionary mapping label values to weights during validation.
            reduction: Type of reduction to apply to the loss.

        Returns:
            The weighted average error for all unique combinations of the samples in the batch.
        """
        batch_size = tf.shape(y_true)[0]
        dtype = tf.float32  # Use high precision for stability
        y_true = tf.cast(y_true, dtype)
        z_pred = tf.cast(z_pred, dtype)
        epsilon = tf.keras.backend.epsilon()

        # Compute pairwise differences using broadcasting
        # For labels: absolute difference (same as original)
        y_diff = tf.abs(y_true - tf.transpose(y_true))
        
        # For representations: cosine distance (1 - dot product) for unit norm vectors
        # Compute pairwise dot products: z_pred[i] · z_pred[j]
        dot_products = tf.matmul(z_pred, tf.transpose(z_pred))
        
        # Cosine distance = 1 - dot_product (for unit norm vectors)
        # Clamp dot products to [-1, 1] to handle numerical errors
        dot_products = tf.clip_by_value(dot_products, -1.0, 1.0)
        z_diff = 1.0 - dot_products
        
        # Ensure z_diff is non-negative (cosine distance should be in [0, 2])
        z_diff = tf.maximum(z_diff, 0.0)

        # Exclude diagonal elements
        num_comparisons = tf.cast(batch_size * (batch_size - 1), dtype)
        y_diff = tf.cast(y_diff, dtype)  # Ensure y_diff is in the same dtype as z_diff

        # Compute the pairwise loss (squared difference between distance matrices)
        pairwise_loss = tf.square(z_diff - y_diff)

        # Create weights matrix
        weights_matrix = tf.ones((batch_size, batch_size), dtype=dtype)

        # Apply sample weights if provided
        sample_weights = train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        if sample_weights is not None:
            weights = create_weight_tensor_fast(y_true, sample_weights)
            # Ensure weights is 1D by squeezing any extra dimensions
            weights = tf.squeeze(weights)
            weights_matrix = tf.cast(weights[:, None] * weights[None, :], dtype)

        # Reshape diagonal zeros to match weights_matrix shape
        diag_zeros = tf.zeros([batch_size], dtype=dtype)
        weights_matrix = tf.linalg.set_diag(weights_matrix, diag_zeros)

        # Apply the weights to the pairwise loss
        pairwise_loss *= weights_matrix

        # Calculate total weighted error
        total_error = tf.reduce_sum(pairwise_loss)

        # Apply reduction
        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error * 0.5  # Account for symmetry in pairs
        elif reduction == tf.keras.losses.Reduction.NONE:
            return total_error / num_comparisons
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def pds_sq_loss_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes the weighted loss for a batch of predicted features and their labels,
        using squared differences for labels and squared L2 norm for representations.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1].
            z_pred: A batch of predicted Z values, shape of [batch_size, n_features].
            phase_manager: Manager that tracks training/validation phase.
            train_sample_weights: Dictionary mapping label values to weights during training.
            val_sample_weights: Dictionary mapping label values to weights during validation.
            reduction: Type of reduction to apply to the loss.

        Returns:
            The weighted average error for all unique combinations of the samples in the batch.
        """
        batch_size = tf.shape(y_true)[0]
        dtype = tf.float32  # Use high precision for stability
        y_true = tf.cast(y_true, dtype)
        z_pred = tf.cast(z_pred, dtype)
        epsilon = tf.keras.backend.epsilon()

        # Compute pairwise differences using broadcasting
        y_diff = tf.square(y_true - tf.transpose(y_true))  # Square instead of abs
        z_diff = tf.reduce_sum(  # No square root, just squared L2 norm
            tf.square(
                z_pred[:, tf.newaxis, :] - z_pred[tf.newaxis, :, :]
            ), axis=-1
        )

        # Exclude diagonal elements
        num_comparisons = tf.cast(batch_size * (batch_size - 1), dtype)
        y_diff = tf.cast(y_diff, dtype)  # Ensure y_diff is in the same dtype as z_diff

        # Compute the pairwise loss
        pairwise_loss = tf.square(z_diff - y_diff)

        # Create weights matrix
        weights_matrix = tf.ones((batch_size, batch_size), dtype=dtype)

        # Apply sample weights if provided
        sample_weights = train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        if sample_weights is not None:
            weights = create_weight_tensor_fast(y_true, sample_weights)
            weights_matrix = tf.cast(weights[:, None] * weights[None, :], dtype)

        # Reshape diagonal zeros to match weights_matrix shape
        diag_zeros = tf.zeros([batch_size], dtype=dtype)
        weights_matrix = tf.linalg.set_diag(weights_matrix, diag_zeros)

        # Apply the weights to the pairwise loss
        pairwise_loss *= weights_matrix

        # Calculate total weighted error
        total_error = tf.reduce_sum(pairwise_loss)

        # Apply reduction
        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error * 0.5  # Account for symmetry in pairs
        elif reduction == tf.keras.losses.Reduction.NONE:
            return total_error / num_comparisons
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def pds_loss_linear_m_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Computes PDS loss using selected pairs of samples:
        - 0 to batch_size - 1 paired with 1 to batch_size.
        - The first element paired with the last element.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1].
            z_pred: A batch of predicted Z values, shape of [batch_size, d].
            phase_manager: Manager that tracks training/validation phase.
            train_sample_weights: Dictionary mapping label values to weights during training.
            val_sample_weights: Dictionary mapping label values to weights during validation.
            reduction: Type of reduction to apply to the loss.

        Returns:
            The weighted average error for the selected combinations of samples in the batch.
        """
        batch_size = tf.shape(y_true)[0]
        dtype = z_pred.dtype
        y_true = tf.cast(y_true, dtype)
        eps = tf.keras.backend.epsilon()

        # Indices for pairwise combinations
        indices1 = tf.range(batch_size - 1)  # 0 -> batch_size - 2
        indices2 = tf.range(1, batch_size)  # 1 -> batch_size - 1
        first_last_indices1 = tf.constant([0], dtype=tf.int32)  # First element
        first_last_indices2 = tf.expand_dims(batch_size - 1, axis=0)  # Last element (Tensor-compatible)

        # Concatenate indices for the two sets of pairs
        indices1 = tf.concat([indices1, first_last_indices1], axis=0)
        indices2 = tf.concat([indices2, first_last_indices2], axis=0)

        # Calculate pairwise differences
        z_pairs_diff = tf.gather(z_pred, indices1) - tf.gather(z_pred, indices2)
        y_pairs_diff = tf.gather(y_true, indices1) - tf.gather(y_true, indices2)

        # Compute L2 norm and absolute differences
        z_diff = tf.sqrt(tf.reduce_sum(tf.square(z_pairs_diff), axis=-1) + eps)
        y_diff = tf.abs(tf.squeeze(y_pairs_diff))

        # Compute pairwise loss
        pairwise_loss = tf.square(z_diff - y_diff)

        # Create sample weights if provided
        sample_weights = (
            train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        )
        if sample_weights:
            weights = tf.concat(
                [
                    create_weight_tensor_fast(tf.gather(y_true, indices1), sample_weights)
                    * create_weight_tensor_fast(tf.gather(y_true, indices2), sample_weights),
                ],
                axis=0,
            )
            pairwise_loss *= tf.cast(weights, dtype=dtype)

        # Calculate total error
        total_error = tf.reduce_sum(pairwise_loss)

        # Apply the requested reduction
        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error
        elif reduction == tf.keras.losses.Reduction.NONE:
            num_pairs = tf.cast(batch_size, dtype=dtype)  # Total number of pairs
            return total_error / num_pairs
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def pds_loss_linear_vec(
            self,
            y_true: tf.Tensor,
            z_pred: tf.Tensor,
            phase_manager: TrainingPhaseManager,
            train_sample_weights: Optional[Dict[float, float]] = None,
            val_sample_weights: Optional[Dict[float, float]] = None,
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    ) -> tf.Tensor:
        """
        Optimized version of PDS loss computation for d-dimensional vectors,
        using absolute differences for labels and L2 norm for representations.

        Args:
            y_true: A batch of true label values, shape of [batch_size, 1].
            z_pred: A batch of predicted Z values, shape of [batch_size, d].
            phase_manager: Manager that tracks training/validation phase.
            train_sample_weights: A dictionary mapping label values to weights during training.
            val_sample_weights: A dictionary mapping label values to weights during validation.
            reduction: Type of reduction to apply to the loss.

        Returns:
            The weighted average error for the unique combinations of samples in the batch.
        """
        batch_size = tf.shape(y_true)[0]
        dtype = z_pred.dtype
        y_true = tf.cast(y_true, dtype)
        eps = tf.keras.backend.epsilon()

        # Compute indices for all pairwise differences
        indices1 = tf.concat([tf.range(batch_size - 1), tf.zeros(batch_size - 2, dtype=tf.int32)], axis=0)
        indices2 = tf.concat([tf.range(1, batch_size), tf.range(2, batch_size)], axis=0)

        # Calculate all pairwise differences at once
        z_pairs_diff = tf.gather(z_pred, indices1) - tf.gather(z_pred, indices2)
        y_pairs_diff = tf.gather(y_true, indices1) - tf.gather(y_true, indices2)

        # Compute L2 norm with epsilon for stability
        z_diff = tf.sqrt(tf.reduce_sum(tf.square(z_pairs_diff), axis=-1) + eps)
        y_diff = tf.abs(tf.squeeze(y_pairs_diff))

        # Compute pairwise loss
        pairwise_loss = tf.square(z_diff - y_diff)

        # Create sample weights if provided
        sample_weights = (
            train_sample_weights if phase_manager.is_training_phase() else val_sample_weights
        )
        if sample_weights:
            weights = tf.concat(
                [
                    create_weight_tensor_fast(tf.gather(y_true, indices1), sample_weights)
                    * create_weight_tensor_fast(tf.gather(y_true, indices2), sample_weights),
                ],
                axis=0,
            )
            # Normalize weights to sum to 1
            # weights = weights / (tf.reduce_sum(weights) + eps)
            pairwise_loss *= tf.cast(weights, dtype=dtype)

        # Calculate total error
        total_error = tf.reduce_sum(pairwise_loss)

        # Apply the requested reduction
        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error
        elif reduction == tf.keras.losses.Reduction.NONE:
            batch_size = tf.cast(batch_size, dtype=dtype)
            return total_error / (2.0 * batch_size - 3.0)
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")

    def update_pair_counts(self, label1, label2):
        label1 = label1[0]
        label2 = label2[0]

        is_sep_1 = label1 > np.log(10.0)
        is_sep_2 = label2 > np.log(10.0)
        is_elevated_1 = (label1 > np.log(10.0 / np.exp(2))) & (label1 <= np.log(10.0))
        is_elevated_2 = (label2 > np.log(10.0 / np.exp(2))) & (label2 <= np.log(10.0))
        is_background_1 = label1 <= np.log(10.0 / np.exp(2))
        is_background_2 = label2 <= np.log(10.0 / np.exp(2))

        self.sep_sep_count.assign_add(tf.cast(is_sep_1 & is_sep_2, tf.int32))
        self.sep_elevated_count.assign_add(tf.cast((is_sep_1 & is_elevated_2) | (is_elevated_1 & is_sep_2), tf.int32))
        self.sep_background_count.assign_add(
            tf.cast((is_sep_1 & is_background_2) | (is_background_1 & is_sep_2), tf.int32))
        self.elevated_elevated_count.assign_add(tf.cast(is_elevated_1 & is_elevated_2, tf.int32))
        self.elevated_background_count.assign_add(
            tf.cast((is_elevated_1 & is_background_2) | (is_background_1 & is_elevated_2), tf.int32))
        self.background_background_count.assign_add(tf.cast(is_background_1 & is_background_2, tf.int32))

    def pds_loss_vec_mgpu(
            self,
            y_true,
            z_pred,
            quadrant,
            sample_weights=None,
            reduction=tf.keras.losses.Reduction.NONE
    ):
        """
        NOTE: only support 4 GPUs like in AI Panthers
        TODO: needs to be fixed and support arbitrary number of even GPUs
        Vectorized computation of the loss for a batch of predicted features and their labels.
        :param y_true: A batch of true label values, shape of [batch_size, 1].
        :param z_pred: A batch of predicted Z values, shape of [batch_size, d].
        :param quadrant: The quadrant index to compute (0, 1, 2, 3).
        :param sample_weights: A dictionary mapping label values to their corresponding reweight.
        :param reduction: The type of reduction to apply to the loss.
        :return: The average error for all unique combinations of the samples in the batch.
        """

        batch_size = tf.shape(y_true)[0]
        half_batch = batch_size // 2

        # if quadrant == 'A':
        #     y_true_i = y_true[:half_batch]
        #     y_true_j = y_true[:half_batch]
        #     z_pred_i = z_pred[:half_batch]
        #     z_pred_j = z_pred[:half_batch]
        # elif quadrant == 'B':
        #     y_true_i = y_true[:half_batch]
        #     y_true_j = y_true[half_batch:]
        #     z_pred_i = z_pred[:half_batch]
        #     z_pred_j = z_pred[half_batch:]
        # elif quadrant == 'C':
        #     y_true_i = y_true[half_batch:]
        #     y_true_j = y_true[:half_batch]
        #     z_pred_i = z_pred[half_batch:]
        #     z_pred_j = z_pred[:half_batch]
        # elif quadrant == 'D':
        #     y_true_i = y_true[half_batch:]
        #     y_true_j = y_true[half_batch:]
        #     z_pred_i = z_pred[half_batch:]
        #     z_pred_j = z_pred[half_batch:]
        # else:
        #     raise ValueError(f"Unsupported quadrant: {quadrant}.")

        # Define slicing functions for each quadrant
        def get_quadrant_0():
            return y_true[:half_batch], y_true[:half_batch], z_pred[:half_batch], z_pred[:half_batch]

        def get_quadrant_1():
            return y_true[:half_batch], y_true[half_batch:], z_pred[:half_batch], z_pred[half_batch:]

        def get_quadrant_2():
            return y_true[half_batch:], y_true[:half_batch], z_pred[half_batch:], z_pred[:half_batch]

        def get_quadrant_3():
            return y_true[half_batch:], y_true[half_batch:], z_pred[half_batch:], z_pred[half_batch:]

        # Use tf.switch_case to select the appropriate quadrant
        y_true_i, y_true_j, z_pred_i, z_pred_j = tf.switch_case(
            quadrant,
            branch_fns={
                0: get_quadrant_0,
                1: get_quadrant_1,
                2: get_quadrant_2,
                3: get_quadrant_3
            },
            default=lambda: (y_true, y_true, z_pred, z_pred)  # Provide a default case to handle unexpected values
        )

        # Compute pairwise differences for z_pred and y_true using broadcasting within the specified quadrant
        y_true_diff = y_true_i[:, tf.newaxis, :] - y_true_j[tf.newaxis, :, :]  # shape: [half_batch, half_batch, 1]
        z_pred_diff = z_pred_i[:, tf.newaxis, :] - z_pred_j[tf.newaxis, :, :]  # shape: [half_batch, half_batch, d]

        # Calculate squared L2 norm for z_pred differences
        z_diff_squared = tf.reduce_sum(tf.square(z_pred_diff), axis=-1)  # shape: [half_batch, half_batch]
        y_diff_squared = tf.squeeze(tf.square(y_true_diff), axis=-1)  # shape: [half_batch, half_batch]
        # Compute the loss for each pair
        pairwise_loss = tf.square(z_diff_squared - y_diff_squared)  # shape: [half_batch, half_batch]

        # Apply sample weights if provided
        if sample_weights is not None:
            # Convert sample_weights keys to strings
            keys = tf.constant(list(map(str, sample_weights.keys())), dtype=tf.string)
            values = tf.constant(list(sample_weights.values()), dtype=tf.float32)
            table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=1.0)
            # Lookup the weights for each y_true value
            weights = table.lookup(tf.as_string(tf.reshape(y_true, [-1])))
            weights_matrix = weights[:, None] * weights[None, :]
            # Cast weights_matrix to the same data type as z_diff_squared
            weights_matrix = tf.cast(weights_matrix, dtype=z_diff_squared.dtype)
            # Apply the weights to the pairwise loss
            pairwise_loss *= weights_matrix

        # Sum over all unique pairs in the quadrant
        total_error = tf.reduce_sum(pairwise_loss)
        # Number of unique comparisons, excluding self-pairs
        num_comparisons = tf.cast(half_batch * (half_batch - 1), dtype=tf.float32)

        if reduction == tf.keras.losses.Reduction.SUM:
            return total_error * 0.5  # upper triangle only
        elif reduction == tf.keras.losses.Reduction.NONE:
            return total_error / num_comparisons
        else:
            raise ValueError(f"Unsupported reduction type: {reduction}.")


# main run

if __name__ == '__main__':
    print("Testing the vectorized loss function...")
    print("WITHOUT SAMPLE WEIGHTS")
    loss_tester = ModelBuilder()
    # Generate dummy data for testing
    np.random.seed(42)  # For reproducibility
    batchsize = 4096
    z_dim = 128
    y_true_dummy = np.random.rand(batchsize, 1).astype(np.float32) - 0.5
    z_pred_dummy = np.random.rand(batchsize, z_dim).astype(np.float32) - 0.5

    # Fabricated data from the table
    fabricated_z = np.array(
        [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]],
        dtype=np.float32)
    fabricated_y = np.array(
        [[-1], [-1], [-1], [0], [0], [0], [1], [1], [1]],
        dtype=np.float32)

    # print a sample of y and z
    # print(f"y_true_dummy: {y_true_dummy[:5]}")
    # print(f"z_pred_dummy: {z_pred_dummy[:5]}")
    #
    print("y_true_dummy shape:", y_true_dummy.shape)
    print("z_pred_dummy shape:", z_pred_dummy.shape)

    # Normalize z_pred_dummy to make it unit vectors
    z_pred_dummy_normalized = tf.linalg.l2_normalize(z_pred_dummy, axis=1)

    # Convert NumPy arrays to TensorFlow tensors
    y_true_tensor = tf.convert_to_tensor(y_true_dummy, dtype=tf.float32)
    z_pred_tensor = tf.convert_to_tensor(z_pred_dummy_normalized, dtype=tf.float32)

    # Verify normalization: The L2 norm of each vector in z_pred_dummy_normalized should be 1
    l2_norms = tf.norm(z_pred_tensor, axis=1)
    print("L2 norms of z_pred_dummy_normalized (should all be 1):", l2_norms.numpy())

    # Check if all norms are 1
    all_norms_one = np.allclose(l2_norms.numpy(), 1.0)
    print("All vectors are unit vectors:", all_norms_one)
    #
    # # # Convert NumPy arrays to TensorFlow tensors
    # y_true_tensor = tf.convert_to_tensor(y_true_dummy, dtype=tf.float32)
    # z_pred_tensor = tf.convert_to_tensor(z_pred_dummy, dtype=tf.float32)
    # Generate dummy data for testing
    # z_dim = fabricated_z.shape[1]
    #
    # # Print a sample of y and z
    # print(f"fabricated_y: {fabricated_y[:5]}")
    # print(f"fabricated_z: {fabricated_z[:5]}")
    #
    # print("fabricated_y shape:", fabricated_y.shape)
    # print("fabricated_z shape:", fabricated_z.shape)
    #
    # # Convert NumPy arrays to TensorFlow tensors
    # y_true_tensor = tf.convert_to_tensor(fabricated_y, dtype=tf.float32)
    # z_pred_tensor = tf.convert_to_tensor(fabricated_z, dtype=tf.float32)

    # Time and compute loss using the original function
    # print("Computing loss using the original function...")
    # start_time_original = time.time()
    # loss_original = loss_tester.pds_loss(y_true_tensor, z_pred_tensor)
    # end_time_original = time.time()
    # original_duration = end_time_original - start_time_original

    # Time and compute loss using the vectorized function
    print("Computing loss using the vectorized function...")
    start_time_vectorized = time.time()
    loss_vectorized = loss_tester.pds_loss_vec(y_true_tensor, z_pred_tensor)
    end_time_vectorized = time.time()
    vectorized_duration = end_time_vectorized - start_time_vectorized

    # Time and compute loss using the vectorized function
    print("Computing loss using the unit vectorized function...")
    start_time_unit_vectorized = time.time()
    loss_unit_vectorized = loss_tester.pds_loss_unit_vec(y_true_tensor, z_pred_tensor)
    end_time_unit_vectorized = time.time()
    unit_vectorized_duration = end_time_unit_vectorized - start_time_unit_vectorized

    # Time and compute loss using the olin function
    # print("Computing loss using the olin function...")
    # start_time_olin = time.time()
    # loss_olin = loss_tester.pds_olin_loss(y_true_tensor, z_pred_tensor)
    # end_time_olin = time.time()
    # olin_duration = end_time_olin - start_time_olin

    # Time and compute loss using the vectorized olin function
    # print("Computing loss using the vectorized olin function...")
    # start_time_olin_vec = time.time()
    # loss_olin_vec = loss_tester.pds_olin_loss_vec(y_true_tensor, z_pred_tensor)
    # end_time_olin_vec = time.time()
    # olin_vec_duration = end_time_olin_vec - start_time_olin_vec

    # Evaluate the TensorFlow tensors to get their numpy values
    # loss_original_value = loss_original.numpy()
    loss_vectorized_value = loss_vectorized.numpy()
    loss_unit_vectorized_value = loss_unit_vectorized.numpy()
    # loss_olin_value = loss_olin.numpy()
    # loss_olin_vec_value = loss_olin_vec.numpy()

    # Print the losses and timing for comparison
    # print(f"Original Loss: {loss_original_value}, Time Taken: {original_duration} seconds")
    print(f"Vectorized Loss: {loss_vectorized_value}, Time Taken: {vectorized_duration} seconds")
    print(f"Unit Vectorized Loss: {loss_unit_vectorized_value}, Time Taken: {unit_vectorized_duration} seconds")
    # print(f"Olin Loss: {loss_olin_value}, Time Taken: {olin_duration} seconds")
    # print(f"Vectorized Olin Loss: {loss_olin_vec_value}, Time Taken: {olin_vec_duration} seconds")

    # # Check if the losses are approximately equal
    # np.testing.assert_almost_equal(loss_original_value, loss_vectorized_value, decimal=5)
    # print("Test passed: The original and vectorized loss functions return approximately the same value.")

    # Compare the execution time
    # if vectorized_duration < original_duration:
    #     print(f"The vectorized function is faster by {original_duration - vectorized_duration} seconds.")
    # else:
    #     print(f"The original function is faster by {vectorized_duration - original_duration} seconds.")

    # print("WITH SAMPLE WEIGHTS")
    # # Generate dummy data for testing
    # np.random.seed(42)  # For reproducibility
    # batch_size = 200
    # z_dim = 9
    # num_unique_pairs = batch_size * (batch_size - 1) // 2
    # y_true_dummy = np.random.rand(batch_size, 1).astype(np.float32)
    # z_pred_dummy = np.random.rand(batch_size, z_dim).astype(np.float32)
    # sample_weights_dummy = np.random.rand(num_unique_pairs, 1).astype(np.float32)
    #
    # print("y_true_dummy shape:", y_true_dummy.shape)
    # print("z_pred_dummy shape:", z_pred_dummy.shape)
    # print("sample_weights_dummy shape:", sample_weights_dummy.shape)
    #
    # # Convert NumPy arrays to TensorFlow tensors
    # y_true_tensor = tf.convert_to_tensor(y_true_dummy, dtype=tf.float32)
    # z_pred_tensor = tf.convert_to_tensor(z_pred_dummy, dtype=tf.float32)
    # sample_weights_tensor = tf.convert_to_tensor(sample_weights_dummy, dtype=tf.float32)  # Convert sample weights
    #
    # # Time and compute loss using the original function with sample weights
    # print("Computing loss using the original function with sample weights...")
    # start_time_original = time.time()
    # loss_original = loss_tester.pds_loss_dl(y_true_tensor, z_pred_tensor, sample_weights=sample_weights_tensor)
    # end_time_original = time.time()
    # original_duration = end_time_original - start_time_original
    #
    # # Time and compute loss using the vectorized function with sample weights
    # print("Computing loss using the vectorized function with sample weights...")
    # start_time_vectorized = time.time()
    # loss_vectorized = loss_tester.pds_loss_vec(y_true_tensor, z_pred_tensor, sample_weights=sample_weights_tensor)
    # end_time_vectorized = time.time()
    # vectorized_duration = end_time_vectorized - start_time_vectorized
    #
    # # Evaluate the TensorFlow tensors to get their numpy values
    # loss_original_value = loss_original.numpy()
    # loss_vectorized_value = loss_vectorized.numpy()
    #
    # # Print the losses and timing for comparison
    # print(f"Original Loss with Sample Weights: {loss_original_value}, Time Taken: {original_duration} seconds")
    # print(f"Vectorized Loss with Sample Weights: {loss_vectorized_value}, Time Taken: {vectorized_duration} seconds")
    #
    # # Compare the execution time
    # if vectorized_duration < original_duration:
    #     print(
    #         f"The vectorized function with sample weights is faster by {original_duration - vectorized_duration} seconds.")
    # else:
    #     print(
    #         f"The original function with sample weights is faster by {vectorized_duration - original_duration} seconds.")
