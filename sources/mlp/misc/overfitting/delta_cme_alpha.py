import os
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    evaluate_mae,
    evaluate_model_cond,
    process_sep_events,
    get_loss,
    reshape_X,
    plot_error_hist,
    filter_ds,
    stratified_split)
from modules.evaluate.utils import investigate_tsne_delta


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for cme_speed_threshold in [0]:
            for alpha in [0, 0.38, 0.6]:
                for add_slope in [False, True]:
                    # PARAMS
                    # inputs_to_use = ['e0.5']
                    # add_slope = True
                    outputs_to_use = ['delta_p']

                    # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                    inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                    # Construct the title
                    title = f'MLP_{inputs_str}_alpha{alpha:.2f}'

                    # Replace any other characters that are not suitable for filenames (if any)
                    title = title.replace(' ', '_').replace(':', '_')

                    # Create a unique experiment name with a timestamp
                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                    experiment_name = f'{title}_{current_time}'

                    # Set the early stopping patience and learning rate as variables
                    seed = 456789
                    tf.random.set_seed(seed)
                    np.random.seed(seed)
                    # patience = 2000  # higher patience
                    learning_rate = 1e-2  # og learning rate

                    reduce_lr_on_plateau = ReduceLROnPlateau(
                        monitor='loss',
                        factor=0.5,
                        patience=1000,
                        verbose=1,
                        min_delta=1e-5,
                        min_lr=1e-10)

                    weight_decay = 1e-8  # higher weight decay
                    momentum_beta1 = 0.9  # higher momentum beta1
                    batch_size = 10000
                    epochs = 30000  # higher epochs
                    hiddens = [
                        2048, 1024,
                        2048, 1024,
                        1024, 512,
                        1024, 512,
                        512, 256,
                        512, 256,
                        256, 128,
                        256, 128,
                        256, 128,
                        128, 128,
                        128, 128,
                        128, 128
                    ]
                    hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                    loss_key = 'mse'
                    target_change = ('delta_p' in outputs_to_use)
                    alpha_rw = alpha
                    bandwidth = 4.42e-2
                    embed_dim = 128
                    output_dim = len(outputs_to_use)
                    dropout = 0.5
                    activation = None
                    norm = 'batch_norm'
                    cme_speed_threshold = cme_speed_threshold
                    residual = True
                    skipped_layers = 2
                    N = 500
                    lower_threshold = -0.5
                    upper_threshold = 0.5

                    # Initialize wandb
                    wandb.init(project="nasa-ts-delta-overfit", name=experiment_name, config={
                        "inputs_to_use": inputs_to_use,
                        "add_slope": add_slope,
                        # "patience": patience,
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
                        "alpha_rw": alpha_rw,
                        "bandwidth": bandwidth,
                        "embed_dim": embed_dim,
                        "dropout": dropout,
                        "activation": 'LeakyReLU',
                        "norm": norm,
                        'optimizer': 'adamw',
                        'output_dim': output_dim,
                        'architecture': 'mlp',
                        'cme_speed_threshold': cme_speed_threshold,
                        'residual': residual,
                        'skipped_layers': skipped_layers,
                        'ds_version': 5
                    })

                    # set the root directory
                    root_dir = 'data/electron_cme_data_split_v5'
                    # build the dataset
                    X_train, y_train = build_dataset(
                        root_dir + '/training',
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        cme_speed_threshold=cme_speed_threshold
                    )
                    X_test, y_test = build_dataset(
                        root_dir + '/testing',
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        cme_speed_threshold=cme_speed_threshold
                    )
                    X_train_filtered, y_train_filtered = filter_ds(
                        X_train, y_train,
                        low_threshold=lower_threshold,
                        high_threshold=upper_threshold,
                        N=N, seed=seed)
                    X_test_filtered, y_test_filtered = filter_ds(
                        X_test, y_test,
                        low_threshold=lower_threshold,
                        high_threshold=upper_threshold,
                        N=N, seed=seed)

                    # print all cme_files shapes
                    print(f'X_train.shape: {X_train.shape}')
                    print(f'y_train.shape: {y_train.shape}')
                    print(f'X_test.shape: {X_test.shape}')
                    print(f'y_test.shape: {y_test.shape}')

                    # get the number of features
                    n_features = X_train.shape[1]
                    print(f'n_features: {n_features}')

                    # Compute the sample weights
                    delta_train = y_train[:, 0]
                    print(f'delta_train.shape: {delta_train.shape}')

                    print(f'rebalancing the training set...')
                    min_norm_weight = 0.01 / len(delta_train)
                    y_train_weights = exDenseReweights(
                        X_train, delta_train,
                        alpha=alpha_rw, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).reweights
                    print(f'training set rebalanced.')

                    # create the model
                    model_sep = create_mlp(
                        input_dim=n_features,
                        hiddens=hiddens,
                        embed_dim=embed_dim,
                        output_dim=output_dim,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        skipped_layers=skipped_layers
                    )
                    model_sep.summary()

                    X_train = reshape_X(
                        X_train,
                        [n_features],
                        inputs_to_use,
                        add_slope,
                        model_sep.name)

                    X_test = reshape_X(
                        X_test,
                        [n_features],
                        inputs_to_use,
                        add_slope,
                        model_sep.name)

                    # Determine the optimal number of epochs from early stopping
                    final_model_sep = create_mlp(
                        input_dim=n_features,
                        hiddens=hiddens,
                        embed_dim=embed_dim,
                        output_dim=output_dim,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        skipped_layers=skipped_layers
                    )

                    # Recreate the model architecture
                    final_model_sep.compile(
                        optimizer=AdamW(
                            learning_rate=learning_rate,
                            beta_1=momentum_beta1,
                            weight_decay=weight_decay
                        ),
                        loss={'forecast_head': get_loss(loss_key)}
                    )
                    # Train on the full dataset
                    final_model_sep.fit(
                        X_train,
                        {'forecast_head': y_train},
                        sample_weight=y_train_weights,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[reduce_lr_on_plateau, WandbCallback(save_model=False)],
                        verbose=1)

                    # evaluate the model on test cme_files
                    error_mae = evaluate_mae(final_model_sep, X_test, y_test)
                    print(f'mae error: {error_mae}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error": error_mae})

                    # evaluate the model on training cme_files
                    error_mae_train = evaluate_mae(final_model_sep, X_train, y_train)
                    print(f'mae error train: {error_mae_train}')
                    # Log the MAE error to wandb
                    wandb.log({"train_mae_error": error_mae_train})

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/testing'
                    filenames = process_sep_events(
                        test_directory,
                        final_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True,
                        using_cme=True,
                        cme_speed_threshold=cme_speed_threshold)

                    # Log the plot to wandb
                    for filename in filenames:
                        log_title = os.path.basename(filename)
                        wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/training'
                    filenames = process_sep_events(
                        test_directory,
                        final_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True,
                        prefix='training',
                        using_cme=True,
                        cme_speed_threshold=cme_speed_threshold)

                    # Log the plot to wandb
                    for filename in filenames:
                        log_title = os.path.basename(filename)
                        wandb.log({f'training_{log_title}': wandb.Image(filename)})

                    # evaluate the model on test cme_files
                    above_threshold = 0.1
                    error_mae_cond = evaluate_model_cond(
                        final_model_sep, X_test, y_test, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 test: {error_mae_cond}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error_cond_test": error_mae_cond})

                    # evaluate the model on training cme_files
                    error_mae_cond_train = evaluate_model_cond(
                        final_model_sep, X_train, y_train, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 train: {error_mae_cond_train}')
                    # Log the MAE error to wandb

                    filename = plot_error_hist(
                        final_model_sep,
                        X_train, y_train,
                        sample_weights=None,
                        title=title,
                        prefix='training')
                    wandb.log({"training_error_hist": wandb.Image(filename)})

                    filename = plot_error_hist(
                        final_model_sep,
                        X_train, y_train,
                        sample_weights=y_train_weights,
                        title=title,
                        prefix='training_weighted')
                    wandb.log({"training_weighted_error_hist": wandb.Image(filename)})

                    ## Log t-SNE plot
                    # Log the training t-SNE plot to wandb
                    stage1_file_path = investigate_tsne_delta(
                        final_model_sep,
                        X_train_filtered, y_train_filtered, title,
                        'training',
                        model_type='features_reg',
                        save_tag=current_time, seed=seed)
                    wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Log the testing t-SNE plot to wandb
                    stage1_file_path = investigate_tsne_delta(
                        final_model_sep,
                        X_test_filtered, y_test_filtered, title,
                        'testing',
                        model_type='features_reg',
                        save_tag=current_time, seed=seed)
                    wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Finish the wandb run
                    wandb.finish()


if __name__ == '__main__':
    main()
