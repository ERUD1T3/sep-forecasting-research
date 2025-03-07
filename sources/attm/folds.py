import os
from datetime import datetime

import numpy as np
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_delta, plot_repr_corr_dist
from modules.reweighting.ImportanceWeighting import exDenseReweightsD
from modules.shared.globals import *
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    stratified_batch_dataset,
    set_seed,
    cmse,
    filter_ds,
    load_stratified_folds,
)
from sources.attm.modules import create_attentive_model2_dict


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS:
            for rho in ATTM_RHO:
                # PARAMS
                inputs_to_use = INPUTS_TO_USE[0] # Use first input configuration
                add_slope = ADD_SLOPE[0] # Use first add_slope value
                cme_speed_threshold = CME_SPEED_THRESHOLD[0] # Use first threshold value
                outputs_to_use = OUTPUTS_TO_USE
                lambda_factor = LAMBDA_FACTOR_ATTM  # lambda for the loss
                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'attm_amse{alpha_mse:.2f}_v8'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = ATTM_PATIENCE  # higher patience
                learning_rate = ATTM_START_LR  # higher learning rate
                asym_type = ASYM_TYPE

                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor=LR_CB_MONITOR,
                    factor=ATTM_LR_CB_FACTOR,
                    patience=ATTM_LR_CB_PATIENCE,
                    verbose=VERBOSE,
                    min_delta=LR_CB_MIN_DELTA,
                    min_lr=ATTM_LR_CB_MIN_LR)

                weight_decay = ATTM_WD  # higher weight decay
                momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                batch_size = BATCH_SIZE  # higher batch size
                epochs = EPOCHS  # higher epochs
                attn_hiddens = ATTN_HIDDENS
                blocks_hiddens = BLOCKS_HIDDENS

                attn_hiddens_str = (", ".join(map(str, attn_hiddens))).replace(', ', '_')
                blocks_hiddens_str = (", ".join(map(str, blocks_hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = len(outputs_to_use)
                attn_dropout = ATTN_DROPOUT
                attm_dropout = ATTM_DROPOUT  # TODO: review if attm dropout should be different
                activation = ATTM_ACTIVATION
                attn_norm = ATTN_NORM
                attm_norm = ATTM_NORM
                attn_skipped_layers = ATTN_SKIPPED_LAYERS
                attm_skipped_blocks = ATTM_SKIPPED_BLOCKS
                N = N_FILTERED  # number of samples to keep outside the threshold
                lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                mae_plus_threshold = MAE_PLUS_THRESHOLD
                smoothing_method = SMOOTHING_METHOD
                window_size = ATTM_WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = ATTM_VAL_WINDOW_SIZE  # allows margin of error of 10 epochs

                # Initialize wandb
                wandb.init(project="Jan-Report", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "momentum_beta1": momentum_beta1,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    # hidden in a more readable format  (wandb does not support lists)
                    "attn_hiddens": attn_hiddens_str,
                    "blocks_hiddens": blocks_hiddens_str,
                    "loss": 'mse_pcc',
                    "lambda": lambda_factor,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "alphaV_mse": alphaV_mse,
                    "alpha_pcc": alpha_pcc,
                    "alphaV_pcc": alphaV_pcc,
                    "bandwidth": bandwidth,
                    "embed_dim": embed_dim,
                    "attm_dropout": attm_dropout,
                    "attn_dropout": attn_dropout,
                    "activation": 'LeakyReLU',
                    "attn_norm": attn_norm,
                    "attm_norm": attm_norm,
                    'optimizer': 'adamw',
                    'output_dim': output_dim,
                    'architecture': 'attm',
                    'cme_speed_threshold': cme_speed_threshold,
                    'attn_skipped_layers': attn_skipped_layers,
                    'attm_skipped_blocks': attm_skipped_blocks,
                    'ds_version': DS_VERSION,
                    'mae_plus_th': mae_plus_threshold,
                    'sam_rho': rho,
                    'smoothing_method': smoothing_method,
                    'window_size': window_size,
                    'val_window_size': val_window_size,
                    'attm_lr_cb_min_lr': ATTM_LR_CB_MIN_LR,
                    'attm_lr_cb_factor': ATTM_LR_CB_FACTOR,
                    'attm_lr_cb_patience': ATTM_LR_CB_PATIENCE,
                    'asym_type': asym_type
                })

                # set the root directory
                root_dir = DS_PATH
                # build the dataset
                X_train, y_train, logI_train, logI_prev_train = build_dataset(
                    root_dir + '/training',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold,
                    shuffle_data=True)
                # print the training set shapes
                print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
                # getting the reweights for training set
                delta_train = y_train[:, 0]
                print(f'delta_train.shape: {delta_train.shape}')
                print(f'rebalancing the training set...')
                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                mse_train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_mse, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                pcc_train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_pcc, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                print(f'training set rebalanced.')
                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                X_test, y_test, logI_test, logI_prev_test = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)
                # print the test set shapes
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                # filtering training and test sets for additional results
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

                # Create initial model and save weights
                set_seed(seed)
                initial_model = create_attentive_model2_dict(
                    input_dim=n_features,
                    output_dim=output_dim,
                    hidden_blocks=blocks_hiddens,
                    attn_hidden_units=attn_hiddens,
                    attn_hidden_activation=activation,
                    attn_skipped_layers=attn_skipped_layers,
                    skipped_blocks=attm_skipped_blocks,
                    attn_dropout=attn_dropout,
                    dropout=attm_dropout,
                    attn_norm=attn_norm,
                    norm=attm_norm,
                    embed_dim=embed_dim,
                    activation=activation,
                    sam_rho=rho
                )
                initial_model.summary()
                initial_weights = initial_model.get_weights()

                # 4-fold cross-validation
                folds_optimal_epochs = []
                for fold_idx, (X_subtrain, y_subtrain, X_val, y_val) in enumerate(
                        load_stratified_folds(
                            root_dir,
                            inputs_to_use=inputs_to_use,
                            add_slope=add_slope,
                            outputs_to_use=outputs_to_use,
                            cme_speed_threshold=cme_speed_threshold,
                            seed=seed, shuffle=True
                        )
                ):
                    print(f'Fold: {fold_idx}')

                    # print all cme_files shapes
                    print(f'X_subtrain.shape: {X_subtrain.shape}, y_subtrain.shape: {y_subtrain.shape}')
                    print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

                    # Compute the sample weights for subtraining
                    delta_subtrain = y_subtrain[:, 0]
                    print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                    print(f'rebalancing the subtraining set...')
                    min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_subtrain)
                    mse_subtrain_weights_dict = exDenseReweightsD(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_mse, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).label_reweight_dict
                    pcc_subtrain_weights_dict = exDenseReweightsD(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_pcc, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).label_reweight_dict
                    print(f'subtraining set rebalanced.')

                    # Compute the sample weights for validation
                    delta_val = y_val[:, 0]
                    print(f'delta_val.shape: {delta_val.shape}')
                    print(f'rebalancing the validation set...')
                    min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_val)
                    mse_val_weights_dict = exDenseReweightsD(
                        X_val, delta_val,
                        alpha=alphaV_mse, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).label_reweight_dict
                    pcc_val_weights_dict = exDenseReweightsD(
                        X_val, delta_val,
                        alpha=alphaV_pcc, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).label_reweight_dict
                    print(f'validation set rebalanced.')

                    set_seed(seed)

                    # Reset model to initial weights
                    initial_model.set_weights(initial_weights)

                    # Define the EarlyStopping callback
                    early_stopping = SmoothEarlyStopping(
                        monitor=CVRG_METRIC,
                        min_delta=ATTM_CVRG_MIN_DELTA,
                        patience=patience,
                        verbose=VERBOSE,
                        restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                        smoothing_method=smoothing_method,  # 'moving_average'
                        smoothing_parameters={'window_size': window_size})  # 10

                    # Compile the model with the specified learning rate
                    initial_model.compile(
                        optimizer=AdamW(
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            beta_1=momentum_beta1
                        ),
                        loss={
                            'output': lambda y_true, y_pred: cmse(
                                y_true, y_pred,
                                phase_manager=pm,
                                lambda_factor=lambda_factor,
                                train_mse_weight_dict=mse_subtrain_weights_dict,
                                train_pcc_weight_dict=pcc_subtrain_weights_dict,
                                val_mse_weight_dict=mse_val_weights_dict,
                                val_pcc_weight_dict=pcc_val_weights_dict,
                                asym_type=asym_type
                            )
                        }
                    )

                    # Step 1: Create stratified dataset for the subtraining and validation set
                    subtrain_ds, subtrain_steps = stratified_batch_dataset(
                        X_subtrain, y_subtrain, batch_size)

                    # Map the subtraining dataset to return {'output': y} format
                    subtrain_ds = subtrain_ds.map(lambda x, y: (x, {'output': y}))
                    val_data = (X_val, {'output': y_val})

                    # Train the model with the callback
                    history = initial_model.fit(
                        subtrain_ds,
                        steps_per_epoch=subtrain_steps,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=val_data,
                        callbacks=[
                            early_stopping,
                            reduce_lr_on_plateau,
                            WandbCallback(save_model=WANDB_SAVE_MODEL),
                            IsTraining(pm)
                        ],
                        verbose=VERBOSE
                    )

                    # optimal epoch for fold
                    # Use the quadratic fit function to find the optimal epoch
                    optimal_epoch = find_optimal_epoch_by_smoothing(
                        history.history[ES_CB_MONITOR],
                        smoothing_method=smoothing_method,
                        smoothing_parameters={'window_size': val_window_size},
                        mode='min')
                    folds_optimal_epochs.append(optimal_epoch)
                    # wandb log the fold's optimal
                    print(f'fold_{fold_idx}_best_epoch: {folds_optimal_epochs[-1]}')
                    wandb.log({f'fold_{fold_idx}_best_epoch': folds_optimal_epochs[-1]})

                # determine the optimal number of epochs from the folds
                optimal_epochs = int(np.mean(folds_optimal_epochs))
                print(f'optimal_epochs: {optimal_epochs}')
                wandb.log({'optimal_epochs': optimal_epochs})

                set_seed(seed)
                # Reset model to initial weights for cheat training
                initial_model.set_weights(initial_weights)

                # Cheat training to find test set optimal epoch
                print("Running cheat training with test set validation...")

                # Compute test set weights for validation
                delta_test = y_test[:, 0]
                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                mse_test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_mse, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                pcc_test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_pcc, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict

                initial_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={
                        'output': lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_factor,
                            train_mse_weight_dict=mse_train_weights_dict,
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            val_mse_weight_dict=mse_test_weights_dict,
                            val_pcc_weight_dict=pcc_test_weights_dict,
                            asym_type=asym_type
                        )
                    }
                )

                train_ds, train_steps = stratified_batch_dataset(X_train, y_train, batch_size)
                train_ds = train_ds.map(lambda x, y: (x, {'output': y}))
                test_data = (X_test, {'output': y_test})

                early_stopping_cheat = SmoothEarlyStopping(
                    monitor=CVRG_METRIC,
                    min_delta=ATTM_CVRG_MIN_DELTA,
                    patience=patience,
                    verbose=VERBOSE,
                    restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': window_size})

                cheat_history = initial_model.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=test_data,
                    callbacks=[
                        early_stopping_cheat,
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=False),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                cheat_optimal_epoch = find_optimal_epoch_by_smoothing(
                    cheat_history.history[ES_CB_MONITOR],
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')
                print(f'cheat_optimal_epoch (using test set): {cheat_optimal_epoch}')
                wandb.log({'cheat_optimal_epoch': cheat_optimal_epoch})

                set_seed(seed)
                # Reset model to initial weights for final training
                initial_model.set_weights(initial_weights)

                initial_model.summary()
                # Compile the model with the specified learning rate
                initial_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={
                        'output': lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_factor,
                            train_mse_weight_dict=mse_train_weights_dict,
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            asym_type=asym_type
                        )
                    },
                )

                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Map the training dataset to return {'output': y} format
                train_ds = train_ds.map(lambda x, y: (x, {'output': y}))

                # Train on the full dataset
                initial_model.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=optimal_epochs,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Save the final model
                initial_model.save_weights(f"final_model_weights_{experiment_name}_reg.h5")
                # print where the model weights are saved
                print(f"Model weights are saved in final_model_weights_{experiment_name}_reg.h5")

                # evaluate the model error on test set
                error_mae = evaluate_mae(initial_model, X_test, y_test, use_dict=True)
                print(f'mae error: {error_mae}')
                wandb.log({"mae": error_mae})

                # evaluate the model error on training set
                error_mae_train = evaluate_mae(initial_model, X_train, y_train, use_dict=True)
                print(f'mae error train: {error_mae_train}')
                wandb.log({"train_mae": error_mae_train})

                # evaluate the model correlation on test set
                error_pcc = evaluate_pcc(initial_model, X_test, y_test, use_dict=True)
                print(f'pcc error: {error_pcc}')
                wandb.log({"pcc": error_pcc})

                # evaluate the model correlation on training set
                error_pcc_train = evaluate_pcc(initial_model, X_train, y_train, use_dict=True)
                print(f'pcc error train: {error_pcc_train}')
                wandb.log({"train_pcc": error_pcc_train})

                # evaluate the model correlation on test set based on logI and logI_prev
                error_pcc_logI = evaluate_pcc(initial_model, X_test, y_test, logI_test, logI_prev_test,
                                              use_dict=True)
                print(f'pcc error logI: {error_pcc_logI}')
                wandb.log({"pcc_I": error_pcc_logI})

                # evaluate the model correlation on training set based on logI and logI_prev
                error_pcc_logI_train = evaluate_pcc(initial_model, X_train, y_train, logI_train,
                                                    logI_prev_train, use_dict=True)
                print(f'pcc error logI train: {error_pcc_logI_train}')
                wandb.log({"train_pcc_I": error_pcc_logI_train})

                # evaluate the model on test cme_files
                above_threshold = mae_plus_threshold
                # evaluate the model error for rare samples on test set
                error_mae_cond = evaluate_mae(
                    initial_model, X_test, y_test, above_threshold=above_threshold, use_dict=True)
                print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
                wandb.log({"mae+": error_mae_cond})

                # evaluate the model error for rare samples on training set
                error_mae_cond_train = evaluate_mae(
                    initial_model, X_train, y_train, above_threshold=above_threshold, use_dict=True)
                print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                wandb.log({"train_mae+": error_mae_cond_train})

                # evaluate the model correlation for rare samples on test set
                error_pcc_cond = evaluate_pcc(
                    initial_model, X_test, y_test, above_threshold=above_threshold, use_dict=True)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                wandb.log({"pcc+": error_pcc_cond})

                # evaluate the model correlation for rare samples on training set
                error_pcc_cond_train = evaluate_pcc(
                    initial_model, X_train, y_train, above_threshold=above_threshold, use_dict=True)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
                wandb.log({"train_pcc+": error_pcc_cond_train})

                # evaluate the model correlation for rare samples on test set based on logI and logI_prev
                error_pcc_cond_logI = evaluate_pcc(
                    initial_model, X_test, y_test, logI_test, logI_prev_test,
                    above_threshold=above_threshold, use_dict=True)
                print(f'pcc error delta >= {above_threshold} logI test: {error_pcc_cond_logI}')
                wandb.log({"pcc+_I": error_pcc_cond_logI})

                # evaluate the model correlation for rare samples on training set based on logI and logI_prev
                error_pcc_cond_logI_train = evaluate_pcc(
                    initial_model, X_train, y_train, logI_train, logI_prev_train,
                    above_threshold=above_threshold, use_dict=True)
                print(f'pcc error delta >= {above_threshold} logI train: {error_pcc_cond_logI_train}')
                wandb.log({"train_pcc+_I": error_pcc_cond_logI_train})

                # Process SEP event files in the specified directory
                test_directory = root_dir + '/testing'
                filenames = process_sep_events(
                    test_directory,
                    initial_model,
                    title=title,
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    show_avsp=True,
                    using_cme=True,
                    cme_speed_threshold=cme_speed_threshold,
                    use_dict=True)

                # Log the plot to wandb
                for filename in filenames:
                    log_title = os.path.basename(filename)
                    wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                # Process SEP event files in the specified directory
                test_directory = root_dir + '/training'
                filenames = process_sep_events(
                    test_directory,
                    initial_model,
                    title=title,
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    show_avsp=True,
                    prefix='training',
                    using_cme=True,
                    cme_speed_threshold=cme_speed_threshold,
                    use_dict=True)

                # Log the plot to wandb
                for filename in filenames:
                    log_title = os.path.basename(filename)
                    wandb.log({f'training_{log_title}': wandb.Image(filename)})

                # Evaluate the model correlation with colored
                file_path = plot_repr_corr_dist(
                    initial_model,
                    X_train_filtered, y_train_filtered,
                    title + "_training",
                    model_type='dict'
                )
                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_corr_dist(
                    initial_model,
                    X_test_filtered, y_test_filtered,
                    title + "_test",
                    model_type='dict'
                )
                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Log t-SNE plot
                # Log the training t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    initial_model,
                    X_train_filtered, y_train_filtered, title,
                    'stage2_training',
                    model_type='dict',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Log the testing t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    initial_model,
                    X_test_filtered, y_test_filtered, title,
                    'stage2_testing',
                    model_type='dict',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)
                #
                # # Plot the error histograms
                # filename = plot_error_hist(
                #     final_model_sep,
                #     X_train, y_train,
                #     sample_weights=None,
                #     title=title,
                #     prefix='training')
                # wandb.log({"training_error_hist": wandb.Image(filename)})
                #
                # # Plot the error histograms on the testing set
                # filename = plot_error_hist(
                #     final_model_sep,
                #     X_test, y_test,
                #     sample_weights=None,
                #     title=title,
                #     prefix='testing')
                # wandb.log({"testing_error_hist": wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
