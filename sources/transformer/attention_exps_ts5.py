import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.shared.globals import *
from modules.training.ts_modeling import (
    build_dataset)
from modules.training.ts_modeling import set_seed
# Importing the Blocks
from sources.transformer.modules import *

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

devices = tf.config.list_physical_devices('GPU')
print(f'devices: {devices}')

set_seed(SEEDS[0])  # Set seed for reproducibility

# set the root directory
root_dir = DS_PATH
inputs_to_use = INPUTS_TO_USE[0]
outputs_to_use = OUTPUTS_TO_USE
cme_speed_threshold = CME_SPEED_THRESHOLD[0]
add_slope = ADD_SLOPE[0]
# build the dataset
x_train, y_train = build_dataset(
    root_dir + '/training',
    inputs_to_use=inputs_to_use,
    add_slope=add_slope,
    outputs_to_use=outputs_to_use,
    cme_speed_threshold=cme_speed_threshold,
    shuffle_data=True)

x_test, y_test = build_dataset(
    root_dir + '/testing',
    inputs_to_use=inputs_to_use,
    add_slope=add_slope,
    outputs_to_use=outputs_to_use,
    cme_speed_threshold=cme_speed_threshold)

x_debug, y_debug = build_dataset(
    root_dir + '/debug',
    inputs_to_use=inputs_to_use,
    add_slope=add_slope,
    outputs_to_use=outputs_to_use,
    cme_speed_threshold=cme_speed_threshold)

hiddens = [128]

# Verify data shapes
print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of x_test: {x_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of x_debug: {x_debug.shape}")
print(f"Shape of y_debug: {y_debug.shape}")


def create_model(block_class, input_shape: Tuple[int]) -> Model:
    """
    Create a model using the specified block class.

    Args:
        block_class: The block class to use.
        input_shape (Tuple[int]): The shape of the input tensor.

    Returns:
        Model: The Keras model.
    """
    inputs = Input(shape=input_shape)
    block = block_class(
        attn_hidden_units=hiddens,
        activation='leaky_relu',
        output_activation='linear')
    outputs = block(inputs)  # dict of outputs and attention scores
    model = Model(inputs, outputs=outputs)
    return model


def train_and_print_results(
        block_name: str,
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_debug: np.ndarray = None,
        y_debug: np.ndarray = None,
        learning_rate: float = 0.003,
        weight_decay: float = 1e-8,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int = 100,
        debug: bool = True,
        time: str = "") -> None:
    """
    Train the model and print the results, including learned weights and attention scores.

    Args:
        block_name (str): Name of the attention block.
        model (Model): The Keras model to train.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        x_debug (np.ndarray): Debug features.
        y_debug (np.ndarray): Debug labels.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        patience (int): Patience for early stopping.
        time (str): The current time for the experiment.
    """
    model.compile(
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss={'output': 'mse'},
        metrics={'output': 'mae'}
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        x_train,
        {'output': y_train},
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_test, {'output': y_test}),
        callbacks=[early_stopping, WandbCallback(save_model=False)],
    )

    # Evaluate the model - focus on the 'output' key
    results = model.evaluate(x_test, {'output': y_test}, verbose=0, return_dict=True)
    print('results')
    print(results)
    loss = results['loss']
    mae = results[f'block_t{block_name}_1_mae']
    print(f"Test loss: {loss}")
    print(f"MAE loss: {mae}")
    # log the results
    wandb.log({"loss": loss, "mae": mae})

    # Evaluate the model on the debug set
    results = model.evaluate(x_debug, {'output': y_debug}, verbose=0, return_dict=True)
    print('results')
    print(results)
    loss = results['loss']
    mae = results[f'block_t{block_name}_1_mae']
    print(f"Debug loss: {loss}")
    print(f"Debug MAE loss: {mae}")
    # log the results
    wandb.log({"debug_loss": loss, "debug_mae": mae})

    if debug:
        # Predict on initial data
        predictions = model.predict(x_debug)
        output_predictions = predictions['output']
        attention_scores = predictions['attention_scores']
        print("Predictions on initial data:")

        if block_name != '2':
            # Retrieve the weights and bias of the last dense layer
            dense_layer_weights, dense_layer_bias = model.layers[-1].get_dense_weights()
            print('weights')
            print(dense_layer_weights)
            print(dense_layer_bias)
        else:
            dense_layer_weights, dense_layer_bias = np.array([[1], [1]]), np.array([0])

        results = []
        for pred, true, inp, attn in zip(output_predictions, y_debug, x_debug, attention_scores):
            attention_weighted_values = [a * w for a, w in zip(attn, dense_layer_weights[:, 0])]
            results.append(
                list(inp) + [true, pred[0]]
                + attn.tolist()
                + attention_weighted_values
                + [dense_layer_bias[0]]
                + dense_layer_weights[:, 0].tolist()
            )

        # Print results in a table
        headers = (
                [f'x{i + 1}' for i in range(len(inp))]
                + ['True y', 'Predicted y']
                + [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])]
                + [f'Attention_Weight_{i + 1}' for i in range(attention_scores.shape[1])]
                + ['Bias'] + [f'Weight_{i + 1}' for i in range(dense_layer_weights.shape[0])]
        )

        df_results = pd.DataFrame(results, columns=headers)
        print(df_results)
        wandb.log({f"results_{block_name}_{time}": df_results})  # so cool you can log dataframes


# Training and printing results for each attention type
input_shape = x_train.shape[1]
block_classes = [BlockT0, BlockT1, BlockT2, BlockT3, BlockT4, BlockT5, BlockT6, BlockT7]

for i, block_class in enumerate(block_classes):
    if i not in [7]:
        continue  # skip the first 4
    # Create a unique experiment name with a timestamp
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f'Attention_Type{i}_{current_time}'
    # Initialize wandb
    LR = 3e-3
    EPOCHS = int(50e3)
    BS = 4096
    PATIENCE = 1000

    wandb.init(project="attention-exps-5", name=experiment_name, config={
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BS,
        "attention_type": i,
        "patience": PATIENCE,
        'attn_hiddens': (", ".join(map(str, hiddens))).replace(', ', '_')
    })

    print(f"\nAttention Type {i}")
    model = create_model(block_class, input_shape)
    model.summary()
    # tf.keras.utils.plot_model(model, to_file=f'./model_{i}.png', show_shapes=True)
    train_and_print_results(
        str(i), model,
        x_train, y_train,
        x_test, y_test,
        x_debug=x_debug, y_debug=y_debug,
        learning_rate=LR, epochs=EPOCHS,
        batch_size=BS, patience=PATIENCE,
        time=current_time
    )

    # Finish the wandb run
    wandb.finish()