import os
from datetime import datetime

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
import numpy as np

from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp_moe,
    evaluate_model,
    evaluate_model_cond,
    process_sep_events,
    get_loss,
    filter_ds, stratified_split, plot_error_hist)

from modules.shared.globals import *


def main():
    """
    Main function to run the E-MLP model
    :return:
    """
    for seed in SEEDS:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha in [0.5]:
                    for add_slope in ADD_SLOPE:
                        # PARAMS
                        outputs_to_use = OUTPUTS_TO_USE

                        # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                        inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                        # Construct the title
                        title = f'Inves_MLP__{inputs_str}_slope{str(add_slope)}_alpha{alpha:.2f}_CME{cme_speed_threshold}'

                        # Replace any other characters that are not suitable for filenames (if any)
                        title = title.replace(' ', '_').replace(':', '_')

                        # Create a unique experiment name with a timestamp
                        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                        experiment_name = f'{title}_{current_time}'

                        # Set the early stopping patience and learning rate as variables
                        tf.random.set_seed(seed)
                        np.random.seed(seed)
                        patience = PATIENCE  # higher patience
                        learning_rate = START_LR_FT

                        reduce_lr_on_plateau = ReduceLROnPlateau(
                            monitor=LR_CB_MONITOR,
                            factor=LR_CB_FACTOR,
                            patience=LR_CB_PATIENCE,
                            verbose=VERBOSE,
                            min_delta=LR_CB_MIN_DELTA,
                            min_lr=LR_CB_MIN_LR)

                        weight_decay = WEIGHT_DECAY  # higher weight decay
                        momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                        batch_size = BATCH_SIZE  # higher batch size
                        epochs = EPOCHS  # higher epochs
                        hiddens =  MLP_HIDDENS  # Hidden layers
                        hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                        loss_key = LOSS_KEY
                        target_change = ('delta_p' in outputs_to_use)
                        alpha_rw = alpha
                        bandwidth = BANDWIDTH
                        repr_dim = REPR_DIM
                        output_dim = len(outputs_to_use)
                        dropout = DROPOUT
                        activation = ACTIVATION
                        norm = NORM
                        cme_speed_threshold = cme_speed_threshold
                        residual = RESIDUAL
                        skipped_layers = SKIPPED_LAYERS
                        N = N_FILTERED  # number of samples to keep outside the threshold
                        lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                        upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                        mae_plus_threshold = MAE_PLUS_THRESHOLD

                        # Initialize wandb
                        wandb.init(project="nasa-ts-delta-v6", name=experiment_name, config={
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
                            "seed": seed,
                            "alpha_rw": alpha_rw,
                            "bandwidth": bandwidth,
                            "reciprocal_reweight": RECIPROCAL_WEIGHTS,
                            "repr_dim": repr_dim,
                            "dropout": dropout,
                            "activation": 'LeakyReLU',
                            "norm": norm,
                            'optimizer': 'adamw',
                            'output_dim': output_dim,
                            'architecture': 'mlp',
                            'cme_speed_threshold': cme_speed_threshold,
                            'residual': residual,
                            'skipped_layers': skipped_layers,
                            'ds_version': DS_VERSION,
                            'mae_plus_th': mae_plus_threshold
                        })

                        # set the root directory
                        root_dir = DS_PATH
                        # build the dataset
                        X_train, y_train = build_dataset(
                            root_dir + '/training',
                            inputs_to_use=inputs_to_use,
                            add_slope=add_slope,
                            outputs_to_use=outputs_to_use,
                            cme_speed_threshold=cme_speed_threshold,
                            shuffle_data=True)

                        X_test, y_test = build_dataset(
                            root_dir + '/testing',
                            inputs_to_use=inputs_to_use,
                            add_slope=add_slope,
                            outputs_to_use=outputs_to_use,
                            cme_speed_threshold=cme_speed_threshold)

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

                        # X_subtrain, y_subtrain, X_val, y_val = stratified_split(
                        #     X_train,
                        #     y_train,
                        #     shuffle=True,
                        #     seed=seed,
                        #     split=VAL_SPLIT,
                        #     debug=False)

                        # print all cme_files shapes
                        print(f'X_train.shape: {X_train.shape}')
                        print(f'y_train.shape: {y_train.shape}')
                        # print(f'X_subtrain.shape: {X_subtrain.shape}')
                        # print(f'y_subtrain.shape: {y_subtrain.shape}')
                        print(f'X_test.shape: {X_test.shape}')
                        print(f'y_test.shape: {y_test.shape}')
                        # print(f'X_val.shape: {X_val.shape}')
                        # print(f'y_val.shape: {y_val.shape}')

                        # Compute the sample weights
                        delta_train = y_train[:, 0]
                        # delta_subtrain = y_subtrain[:, 0]
                        # delta_val = y_val[:, 0]
                        delta_test = y_test[:, 0]
                        print(f'delta_train.shape: {delta_train.shape}')
                        # print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                        # print(f'delta_val.shape: {delta_val.shape}')
                        print(f'delta_test.shape: {delta_test.shape}')

                        print(f'rebalancing the training set...')
                        min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                        y_train_weights = exDenseReweights(
                            X_train, delta_train,
                            alpha=alpha_rw, bw=bandwidth,
                            min_norm_weight=min_norm_weight,
                            debug=False).reweights
                        print(f'training set rebalanced.')

                        # print(f'rebalancing the subtraining set...')
                        # min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_subtrain)
                        # y_subtrain_weights = exDenseReweights(
                        #     X_subtrain, delta_subtrain,
                        #     alpha=alpha_rw, bw=bandwidth,
                        #     min_norm_weight=min_norm_weight,
                        #     debug=False).reweights
                        # print(f'subtraining set rebalanced.')

                        # print(f'rebalancing the validation set...')
                        # min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_val)
                        # y_val_weights = exDenseReweights(
                        #     X_val, delta_val,
                        #     alpha=COMMON_VAL_ALPHA, bw=bandwidth,
                        #     min_norm_weight=min_norm_weight,
                        #     debug=False).reweights
                        # print(f'validation set rebalanced.')

                        print(f'rebalancing the validation set...')
                        min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                        y_test_weights = exDenseReweights(
                            X_test, delta_test,
                            alpha=COMMON_VAL_ALPHA, bw=bandwidth,
                            min_norm_weight=min_norm_weight,
                            debug=False).reweights
                        print(f'validation set rebalanced.')

                        # get the number of features
                        n_features = X_train.shape[1]
                        print(f'n_features: {n_features}')

                        # Parameters for the mixture of experts model
                        expert_high_path = '/home1/jmoukpe2016/keras-functional-api/inves_model_weights_Inves_MLP__e0_5_e1_8_p_slopeFalse_alpha1.00_CME0_20240701-235026_reg_moe.h5'
                        expert_low_path = '/home1/jmoukpe2016/keras-functional-api/inves_model_weights_Inves_MLP__e0_5_e1_8_p_slopeFalse_alpha0.00_CME0_20240701-235026_reg_moe.h5'
                        router_hiddens = MLP_HIDDENS
                        temperature = 0.1 # Example temperature

                        # Create the MoE model
                        model_sep = create_mlp_moe(
                            input_dim=n_features,
                            hiddens=hiddens,
                            repr_dim=repr_dim,
                            output_dim=output_dim,
                            dropout_rate=dropout,
                            activation=activation,
                            norm=norm,
                            residual=residual,
                            skipped_layers=skipped_layers,
                            expert_high_path=expert_high_path,
                            expert_low_path=expert_low_path,
                            router_hiddens=router_hiddens,
                            freeze_experts=False,  # Change to True if you want to freeze the experts
                            temperature=temperature
                        )
                        model_sep.summary()

                        # Define the EarlyStopping callback
                        early_stopping = EarlyStopping(
                            monitor=ES_CB_MONITOR,
                            patience=patience,
                            verbose=VERBOSE,
                            restore_best_weights=True)

                        best_weights_filepath = f"inves_model_weights_{experiment_name}_reg_moe.h5"
                        model_checkpoint = ModelCheckpoint(
                            filepath=best_weights_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1
                        )

                        # Compile the model with the specified learning rate
                        model_sep.compile(
                            optimizer=AdamW(
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                beta_1=momentum_beta1
                            ),
                            loss={'forecast_head': get_loss(loss_key)}
                        )

                        # Train the model with the callback
                        model_sep.fit(
                            X_train,
                            {'forecast_head': y_train},
                            sample_weight=y_train_weights,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, {'forecast_head': y_test}, y_test_weights),
                            callbacks=[
                                early_stopping,
                                reduce_lr_on_plateau,
                                model_checkpoint,  # Save the best model
                                WandbCallback(save_model=WANDB_SAVE_MODEL)
                            ],
                            verbose=VERBOSE
                        )

                         # Save the final model
                        model_sep.load_weights(best_weights_filepath)
                        print(f"Model weights are saved in {best_weights_filepath}")


                        # evaluate the model on test cme_files
                        error_mae = evaluate_model(model_sep, X_test, y_test)
                        print(f'mae error: {error_mae}')
                        # Log the MAE error to wandb
                        wandb.log({"mae_error": error_mae})

                        # evaluate the model on training cme_files
                        error_mae_train = evaluate_model(model_sep, X_train, y_train)
                        print(f'mae error train: {error_mae_train}')
                        # Log the MAE error to wandb
                        wandb.log({"train_mae_error": error_mae_train})

                        # Process SEP event files in the specified directory
                        test_directory = root_dir + '/testing'
                        filenames = process_sep_events(
                            test_directory,
                            model_sep,
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
                            model_sep,
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
                        above_threshold = mae_plus_threshold
                        error_mae_cond = evaluate_model_cond(
                            model_sep, X_test, y_test, above_threshold=above_threshold)

                        print(f'mae error delta >= 0.1 test: {error_mae_cond}')
                        # Log the MAE error to wandb
                        wandb.log({"mae_error_cond_test": error_mae_cond})

                        # evaluate the model on training cme_files
                        error_mae_cond_train = evaluate_model_cond(
                            model_sep, X_train, y_train, above_threshold=above_threshold)

                        print(f'mae error delta >= 0.1 train: {error_mae_cond_train}')
                        # Log the MAE error to wandb
                        wandb.log({"mae_error_cond_train": error_mae_cond_train})
                        # Log the MAE error to wandb

                        # Evaluate the model correlation with colored
                        file_path = plot_repr_corr_dist(
                            model_sep,
                            X_train_filtered, y_train_filtered,
                            title + "_training",
                            model_type='features_reg'
                        )
                        wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        file_path = plot_repr_corr_dist(
                            model_sep,
                            X_test_filtered, y_test_filtered,
                            title + "_test",
                            model_type='features_reg'
                        )
                        wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        # Log t-SNE plot
                        # Log the training t-SNE plot to wandb
                        stage1_file_path = plot_tsne_delta(
                            model_sep,
                            X_train_filtered, y_train_filtered, title,
                            'stage2_training',
                            model_type='features_reg',
                            save_tag=current_time, seed=seed)
                        wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                        print('stage1_file_path: ' + stage1_file_path)

                        # Log the testing t-SNE plot to wandb
                        stage1_file_path = plot_tsne_delta(
                            model_sep,
                            X_test_filtered, y_test_filtered, title,
                            'stage2_testing',
                            model_type='features_reg',
                            save_tag=current_time, seed=seed)
                        wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                        print('stage1_file_path: ' + stage1_file_path)

                        # Plot the error histograms
                        filename = plot_error_hist(
                            model_sep,
                            X_train, y_train,
                            sample_weights=None,
                            title=title,
                            prefix='training')
                        wandb.log({"training_error_hist": wandb.Image(filename)})

                        # Plot the error weighted histograms
                        filename = plot_error_hist(
                            model_sep,
                            X_train, y_train,
                            sample_weights=y_train_weights,
                            title=title,
                            prefix='training_weighted')
                        wandb.log({"training_weighted_error_hist": wandb.Image(filename)})

                        # Plot the error histograms on the testing set
                        filename = plot_error_hist(
                            model_sep,
                            X_test, y_test,
                            sample_weights=None,
                            title=title,
                            prefix='testing')
                        wandb.log({"testing_error_hist": wandb.Image(filename)})

                        # Finish the wandb run
                        wandb.finish()


if __name__ == '__main__':
    main()