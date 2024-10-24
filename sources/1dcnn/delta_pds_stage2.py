import os
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
import numpy as np

from modules.evaluate.utils import plot_tsne_delta, plot_tsne_delta
from modules.training.DenseReweights import exDenseReweights
from modules.training.cme_modeling import ModelBuilder
from modules.training.ts_modeling import (
    build_dataset,
    create_1dcnn,
    evaluate_mae,
    evaluate_model_cond,
    process_sep_events,
    get_loss,
    reshape_X)
from modules.training.utils import get_weight_path

mb = ModelBuilder()

# Define the lookup dictionary
weight_paths = {
    True: '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240322'
          '-1510511DCNN_e0_5_e1_8_p_slopeTrue_PDS_bs5000_features.h5',
    False: '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240322'
           '-1615341DCNN_e0_5_e1_8_p_slopeFalse_PDS_bs5000_features.h5',
}


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for add_slope in [True, False]:
            for freeze in [False, True]:
                for alpha in np.arange(0.1, 1.6, 0.25):
                    # PARAMS
                    # inputs_to_use = ['e0.5']
                    # add_slope = True
                    outputs_to_use = ['delta_p']
                    # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                    inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                    # Construct the title
                    title = f'1DCNN_PDS_Stage2_{inputs_str}_frozen{freeze}_alpha{alpha:.2f}'

                    # Replace any other characters that are not suitable for filenames (if any)
                    title = title.replace(' ', '_').replace(':', '_')

                    # Create a unique experiment name with a timestamp
                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                    experiment_name = f'{title}_{current_time}'

                    seed = 456789
                    tf.random.set_seed(seed)
                    np.random.seed(seed)
                    patience = 5000  # higher patience
                    learning_rate = 5e-3  # og learning rate
                    reduce_lr_on_plateau = ReduceLROnPlateau(
                        monitor='loss',
                        factor=0.5,
                        patience=300,
                        verbose=1,
                        min_delta=1e-5,
                        min_lr=1e-10)

                    weight_decay = 1e-6  # higher weight decay
                    momentum_beta1 = 0.9  # higher momentum beta1
                    batch_size = 4096
                    epochs = 50000  # higher epochs
                    hiddens = [
                        (32, 10, 1, 'none', 0),  # Conv1: Start with broad features
                        (64, 8, 1, 'max', 2),  # Conv2 + Pool: Start to reduce and capture features
                        (64, 7, 1, 'none', 0),  # Conv3: Further detail without reducing dimension
                        (128, 5, 1, 'none', 0),  # Conv4: Increase filters, capture more refined features
                        (128, 5, 1, 'max', 2),  # Conv5 + Pool: Reduce dimension and increase depth
                        (256, 3, 1, 'none', 0),  # Conv6: Increase depth without immediate pooling
                        (256, 3, 1, 'none', 0),  # Conv7: Continue with high capacity, no dimension reduction
                        (512, 3, 1, 'max', 2),  # Conv8 + Pool: Use max pooling for stronger feature selection
                        (512, 3, 1, 'none', 0),  # Conv9: Maintain capacity, focusing on detailed features
                        # Note: Assuming a reduction towards dense layers after Conv9
                    ]
                    proj_hiddens = [6]
                    hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                    loss_key = 'mse'
                    target_change = ('delta_p' in outputs_to_use)
                    # print_batch_mse_cb = PrintBatchMSE()
                    rebalacing = True
                    alpha_rw = alpha
                    bandwidth = 0.099  # 0.0519
                    repr_dim = 9
                    output_dim = len(outputs_to_use)
                    dropout = 0.5
                    activation = None
                    norm = 'batch_norm'
                    pds = True
                    weight_path = get_weight_path(weight_paths, add_slope)

                    # Initialize wandb
                    wandb.init(project="nasa-ts-pds-delta-2", name=experiment_name, config={
                        "inputs_to_use": inputs_to_use,
                        "add_slope": add_slope,
                        "patience": patience,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "momentum_beta1": momentum_beta1,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        # hidden in a more readable format  (wandb does not support lists)
                        "hiddens": hiddens_str,
                        "loss": loss_key,
                        "target_change": target_change,
                        "printing_batch_mse": False,
                        "seed": seed,
                        "rebalancing": rebalacing,
                        "alpha_rw": alpha_rw,
                        "bandwidth": bandwidth,
                        "reciprocal_reweight": True,
                        "repr_dim": repr_dim,
                        "dropout": dropout,
                        "activation": 'LeakyReLU',
                        "norm": norm,
                        'optimizer': 'adamw',
                        'output_dim': output_dim,
                        'architecture': '1dcnn',
                        "freeze": freeze,
                        "pds": pds,
                        "stage": 2,
                        "stage1_weights": weight_path
                    })

                    # set the root directory
                    root_dir = 'data/electron_cme_data_split'
                    # build the dataset
                    X_train, y_train = build_dataset(root_dir + '/training',
                                                     inputs_to_use=inputs_to_use,
                                                     add_slope=add_slope,
                                                     outputs_to_use=outputs_to_use)
                    X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining',
                                                           inputs_to_use=inputs_to_use,
                                                           add_slope=add_slope,
                                                           outputs_to_use=outputs_to_use)
                    X_test, y_test = build_dataset(root_dir + '/testing',
                                                   inputs_to_use=inputs_to_use,
                                                   add_slope=add_slope,
                                                   outputs_to_use=outputs_to_use)
                    X_val, y_val = build_dataset(root_dir + '/validation',
                                                 inputs_to_use=inputs_to_use,
                                                 add_slope=add_slope,
                                                 outputs_to_use=outputs_to_use)

                    # print all cme_files shapes
                    print(f'X_train.shape: {X_train.shape}')
                    print(f'y_train.shape: {y_train.shape}')
                    print(f'X_subtrain.shape: {X_subtrain.shape}')
                    print(f'y_subtrain.shape: {y_subtrain.shape}')
                    print(f'X_test.shape: {X_test.shape}')
                    print(f'y_test.shape: {y_test.shape}')
                    print(f'X_val.shape: {X_val.shape}')
                    print(f'y_val.shape: {y_val.shape}')

                    # Compute the sample weights
                    delta_train = y_train[:, 0]
                    delta_subtrain = y_subtrain[:, 0]
                    print(f'delta_train.shape: {delta_train.shape}')
                    print(f'delta_subtrain.shape: {delta_subtrain.shape}')

                    print(f'rebalancing the training set...')
                    min_norm_weight = 0.01 / len(delta_train)
                    y_train_weights = exDenseReweights(
                        X_train, delta_train,
                        alpha=alpha_rw, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).reweights
                    print(f'training set rebalanced.')

                    print(f'rebalancing the subtraining set...')
                    min_norm_weight = 0.01 / len(delta_subtrain)
                    y_subtrain_weights = exDenseReweights(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_rw, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).reweights
                    print(f'subtraining set rebalanced.')

                    # get the number of features
                    if add_slope:
                        # n_features = [25] * len(inputs_to_use) * 2
                        n_features = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
                    else:
                        n_features = [25] * len(inputs_to_use)
                    print(f'n_features: {n_features}')

                    # create the model
                    mlp_model_sep_stage1 = create_1dcnn(
                        input_dims=n_features,
                        hiddens=hiddens,
                        output_dim=0,
                        pds=pds,
                        repr_dim=repr_dim,
                        dropout=dropout,
                        activation=activation,
                        norm=norm
                    )
                    mlp_model_sep_stage1.summary()
                    # load the weights from the first stage

                    print(f'weights loading from: {weight_path}')
                    mlp_model_sep_stage1.load_weights(weight_path)
                    # print the save
                    print(f'weights loaded successfully from: {weight_path}')

                    # Log t-SNE plot for training
                    # Log the training t-SNE plot to wandb
                    stage1_file_path = plot_tsne_delta(mlp_model_sep_stage1, X_train, y_train, title, 'stage1_training',
                                                       save_tag=current_time, seed=seed)
                    wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Log t-SNE plot for testing
                    # Log the testing t-SNE plot to wandb
                    stage1_file_path = plot_tsne_delta(mlp_model_sep_stage1, X_test, y_test, title, 'stage1_testing',
                                                       save_tag=current_time, seed=seed)
                    wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    mlp_model_sep = mb.add_proj_head(
                        mlp_model_sep_stage1,
                        output_dim=output_dim,
                        freeze_features=freeze,
                        pds=pds,
                        hiddens=proj_hiddens,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        name='1dcnn'
                    )
                    mlp_model_sep.summary()

                    print('Reshaping input for model')
                    X_subtrain = reshape_X(
                        X_subtrain,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        '1dcnn')

                    X_val = reshape_X(
                        X_val,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        '1dcnn')

                    X_train = reshape_X(
                        X_train,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        '1dcnn')

                    X_test = reshape_X(
                        X_test,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        '1dcnn')

                    # Define the EarlyStopping callback
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        verbose=1,
                        restore_best_weights=True)

                    # Compile the model with the specified learning rate
                    mlp_model_sep.compile(optimizer=AdamW(learning_rate=learning_rate,
                                                          weight_decay=weight_decay,
                                                          beta_1=momentum_beta1),
                                          loss={'forecast_head': get_loss(loss_key)})

                    # Train the model with the callback
                    history = mlp_model_sep.fit(X_subtrain,
                                                {'forecast_head': y_subtrain},
                                                sample_weight=y_subtrain_weights,
                                                epochs=epochs, batch_size=batch_size,
                                                validation_data=(X_val, {'forecast_head': y_val}),
                                                callbacks=[
                                                    early_stopping,
                                                    WandbCallback(save_model=False),
                                                    reduce_lr_on_plateau
                                                ])

                    # Plot the training and validation loss
                    # plt.figure(figsize=(12, 6))
                    # plt.plot(history.history['loss'], label='Training Loss')
                    # plt.plot(history.history['val_loss'], label='Validation Loss')
                    # plt.title('Training and Validation Loss')
                    # plt.xlabel('Epochs')
                    # plt.ylabel('Loss')
                    # plt.legend()
                    # # save the plot
                    # plt.savefig(f'mlp_loss_{title}.png')

                    # Determine the optimal number of epochs from early stopping
                    optimal_epochs = early_stopping.stopped_epoch  + 1  # Adjust for the offset
                    final_mlp_model_sep_stage1 = create_1dcnn(
                        input_dims=n_features,
                        hiddens=hiddens,
                        output_dim=0,
                        pds=pds,
                        repr_dim=repr_dim,
                        dropout=dropout,
                        activation=activation,
                        norm=norm
                    )
                    final_mlp_model_sep_stage1.load_weights(weight_path)

                    # Recreate the model architecture for final_mlp_model_sep
                    final_mlp_model_sep = mb.add_proj_head(
                        final_mlp_model_sep_stage1,
                        output_dim=output_dim,
                        freeze_features=freeze,
                        pds=pds,
                        hiddens=proj_hiddens,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        name='1dcnn'
                    )

                    final_mlp_model_sep.compile(
                        optimizer=AdamW(learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        beta_1=momentum_beta1),
                        loss={'forecast_head': get_loss(loss_key)})  # Compile the model just like before
                    # Train on the full dataset
                    final_mlp_model_sep.fit(
                        X_train,
                        {'forecast_head': y_train},
                        sample_weight=y_train_weights,
                        epochs=optimal_epochs,
                        batch_size=batch_size,
                        callbacks=[reduce_lr_on_plateau, WandbCallback(save_model=False)],
                        verbose=1)

                    # evaluate the model on test cme_files
                    error_mae = evaluate_mae(final_mlp_model_sep, X_test, y_test)
                    print(f'mae error: {error_mae}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error": error_mae})

                    # evaluate the model on stage2 cme_files
                    error_mae_train = evaluate_mae(final_mlp_model_sep, X_train, y_train)
                    print(f'mae error train: {error_mae_train}')
                    # Log the MAE error to wandb
                    wandb.log({"train_mae_error": error_mae_train})

                    # Log t-SNE plot for training
                    # Log the training t-SNE plot to wandb
                    stage2_file_path = plot_tsne_delta(final_mlp_model_sep, X_train, y_train, title, 'stage2_training',
                                                       save_tag=current_time, seed=seed)
                    wandb.log({'stage2_tsne_training_plot': wandb.Image(stage2_file_path)})
                    print('stage2_file_path: ' + stage2_file_path)

                    # Log t-SNE plot for testing
                    # Log the testing t-SNE plot to wandb
                    stage2_file_path = plot_tsne_delta(final_mlp_model_sep, X_test, y_test, title, 'stage2_testing',
                                                       save_tag=current_time, seed=seed)
                    wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage2_file_path)})
                    print('stage2_file_path: ' + stage2_file_path)

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/testing'
                    filenames = process_sep_events(
                        test_directory,
                        final_mlp_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True)

                    # Log the plot to wandb
                    for filename in filenames:
                        wandb.log({f'testing_{filename}': wandb.Image(filename)})

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/training'
                    filenames = process_sep_events(
                        test_directory,
                        final_mlp_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True,
                        prefix='training')

                    # Log the plot to wandb
                    for filename in filenames:
                        log_title = os.path.basename(filename)
                        wandb.log({f'training_{log_title}': wandb.Image(filename)})

                    # evaluate the model on test cme_files
                    above_threshold = 0.1
                    error_mae_cond = evaluate_model_cond(
                        final_mlp_model_sep, X_test, y_test, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 test: {error_mae_cond}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error_cond_test": error_mae_cond})

                    # evaluate the model on training cme_files
                    error_mae_cond_train = evaluate_model_cond(
                        final_mlp_model_sep, X_train, y_train, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 train: {error_mae_cond_train}')
                    #

                    # Finish the wandb run
                    wandb.finish()


if __name__ == '__main__':
    main()
