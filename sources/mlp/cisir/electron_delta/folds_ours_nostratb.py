import os
from datetime import datetime

import numpy as np
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
from modules.reweighting.ImportanceWeighting import ReciprocalImportance
from modules.shared.globals import *
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    eval_lt_mae,
    eval_lt_pcc,
    process_sep_events,
    stratified_batch_dataset,
    set_seed,
    cmse,
    filter_ds,
    create_mlp,
    plot_error_hist,
    load_stratified_folds,
)



def main():
    """
    Testing WPCC + Reciprocal Importance + Stratified Batching
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    for seed in TRIAL_SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in [(0.85, 0.85, 0.0, 0.0)]:
            for rho in RHO:  # SAM_RHOS:
                inputs_to_use = INPUTS_TO_USE[0]
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]
                add_slope = ADD_SLOPE[0]
                # PARAMS
                outputs_to_use = OUTPUTS_TO_USE
                lambda_factor = 0.0  # lambda for the loss
                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'mlp_amse{alpha_mse:.2f}_apcc{alpha_pcc:.2f}_nostratb'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = PATIENCE  # higher patience
                learning_rate = START_LR  # starting learning rate
                asym_type = ASYM_TYPE
                lr_cb_patience = LR_CB_PATIENCE
                lr_cb_factor = LR_CB_FACTOR
                lr_cb_min_lr = LR_CB_MIN_LR
                lr_cb_min_delta = LR_CB_MIN_DELTA
                cvrg_metric = CVRG_METRIC
                cvrg_min_delta = CVRG_MIN_DELTA 
                normalized_weights = NORMALIZED_WEIGHTS

                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor=LR_CB_MONITOR,
                    factor=lr_cb_factor,
                    patience=lr_cb_patience,
                    verbose=VERBOSE,
                    min_delta=lr_cb_min_delta,
                    min_lr=lr_cb_min_lr)

                weight_decay = WEIGHT_DECAY  
                batch_size = BATCH_SIZE  
                epochs = EPOCHS  
                hiddens = MLP_HIDDENS  
                pretraining = False
                mae_plus_threshold = MAE_PLUS_THRESHOLD

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = len(outputs_to_use)
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                cme_speed_threshold = cme_speed_threshold
                skip_repr = SKIP_REPR
                skipped_layers = SKIPPED_LAYERS
                N = N_FILTERED  # number of samples to keep outside the threshold
                lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                smoothing_method = SMOOTHING_METHOD
                window_size = WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = VAL_WINDOW_SIZE  # allows margin of error of 10 epochs


                # Initialize wandb
                wandb.init(project="NeurIPS-2025-Paper", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    'min_lr': lr_cb_min_lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    # hidden in a more readable format  (wandb does not support lists)
                    "hiddens": hiddens_str,
                    "loss": 'mse_pcc',
                    "lambda": lambda_factor,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "alphaV_mse": alphaV_mse,
                    "alpha_pcc": alpha_pcc,
                    "alphaV_pcc": alphaV_pcc,
                    "bandwidth": bandwidth,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "activation": 'LeakyReLU',
                    "norm": norm,
                    'optimizer': 'adam',
                    'output_dim': output_dim,
                    'architecture': 'mlp_res_repr',
                    'cme_speed_threshold': cme_speed_threshold,
                    'ds_version': DS_VERSION,
                    'sam_rho': rho,
                    'smoothing_method': smoothing_method,
                    'window_size': window_size,
                    'val_window_size': val_window_size,
                    'skip_repr': skip_repr,
                    'asym_type': asym_type,
                    'lr_cb_patience': lr_cb_patience,
                    'lr_cb_factor': lr_cb_factor,
                    'lr_cb_min_lr': lr_cb_min_lr,
                    'lr_cb_min_delta': lr_cb_min_delta,
                    'cvrg_metric': cvrg_metric,
                    'cvrg_min_delta': cvrg_min_delta,
                    'normalized_weights': normalized_weights,
                    'mae_plus_threshold': mae_plus_threshold
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
                mse_train_weights_dict = ReciprocalImportance(
                    X_train, delta_train,
                    alpha=alpha_mse, 
                    bandwidth=bandwidth).label_importance_map
                pcc_train_weights_dict = ReciprocalImportance(
                    X_train, delta_train,
                    alpha=alpha_pcc, 
                    bandwidth=bandwidth).label_importance_map
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
                    mse_subtrain_weights_dict = ReciprocalImportance(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_mse, 
                        bandwidth=bandwidth).label_importance_map
                    pcc_subtrain_weights_dict = ReciprocalImportance(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_pcc, 
                        bandwidth=bandwidth).label_importance_map
                    print(f'subtraining set rebalanced.')

                    # Compute the sample weights for validation
                    delta_val = y_val[:, 0]
                    print(f'delta_val.shape: {delta_val.shape}')
                    print(f'rebalancing the validation set...')
                    mse_val_weights_dict = ReciprocalImportance(
                        X_val, delta_val,
                        alpha=alphaV_mse, 
                        bandwidth=bandwidth).label_importance_map
                    pcc_val_weights_dict = ReciprocalImportance(
                        X_val, delta_val,
                        alpha=alphaV_pcc, 
                        bandwidth=bandwidth).label_importance_map
                    print(f'validation set rebalanced.')

                    # create the model
                    model_sep = create_mlp(
                        input_dim=n_features,
                        hiddens=hiddens,
                        embed_dim=embed_dim,
                        output_dim=output_dim,
                        dropout=dropout,
                        activation=activation,
                        norm=norm,
                        skip_repr=skip_repr,
                        skipped_layers=skipped_layers,
                        pretraining=pretraining,
                        sam_rho=rho,
                        weight_decay=weight_decay
                    )
                    model_sep.summary()

                    # Define the EarlyStopping callback
                    early_stopping = SmoothEarlyStopping(
                        monitor=cvrg_metric,
                        min_delta=cvrg_min_delta,
                        patience=patience,
                        verbose=VERBOSE,
                        restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                        smoothing_method=smoothing_method,  # 'moving_average'
                        smoothing_parameters={'window_size': window_size})  # 10

                    # Compile the model with the specified learning rate
                    model_sep.compile(
                        optimizer=Adam(
                            learning_rate=learning_rate,
                        ),
                        loss={
                            'forecast_head': lambda y_true, y_pred: cmse(
                                y_true, y_pred,
                                phase_manager=pm,
                                lambda_factor=lambda_factor,
                                train_mse_weight_dict=mse_subtrain_weights_dict,
                                train_pcc_weight_dict=pcc_subtrain_weights_dict,
                                val_mse_weight_dict=mse_val_weights_dict,
                                val_pcc_weight_dict=pcc_val_weights_dict,
                                normalized_weights=normalized_weights,
                                asym_type=asym_type
                            )
                        }
                    )


                    # Train the model with the callback
                    history = model_sep.fit(
                        X_subtrain, {'forecast_head': y_subtrain},
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, {'forecast_head': y_val}),
                        callbacks=[
                            early_stopping,
                            reduce_lr_on_plateau,
                            WandbCallback(save_model=WANDB_SAVE_MODEL),
                            IsTraining(pm)
                        ],
                        verbose=VERBOSE
                    )

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

                final_model_sep = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    embed_dim=embed_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    pretraining=pretraining,
                    sam_rho=rho,
                    weight_decay=weight_decay
                )

                # final_model_sep.summary()
                final_model_sep.compile(
                    optimizer=Adam(
                        learning_rate=learning_rate,
                    ),
                    loss={
                        'forecast_head': lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_factor,
                            train_mse_weight_dict=mse_train_weights_dict,
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            normalized_weights=normalized_weights,
                            asym_type=asym_type
                        )
                    },
                )  # Compile the model just like before

                # Train on the full dataset
                final_model_sep.fit(
                    X_train, {'forecast_head': y_train},
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
                final_model_sep.save_weights(f"final_model_weights_{experiment_name}_reg.h5")
                # print where the model weights are saved
                print(f"Model weights are saved in final_model_weights_{experiment_name}_reg.h5")

                # evaluate the model error on test set
                error_mae = evaluate_mae(final_model_sep, X_test, y_test)
                print(f'mae error: {error_mae}')
                wandb.log({"mae": error_mae})

                # evaluate the model error on training set
                error_mae_train = evaluate_mae(final_model_sep, X_train, y_train)
                print(f'mae error train: {error_mae_train}')
                wandb.log({"train_mae": error_mae_train})

                # evaluate the model correlation on test set
                error_pcc = evaluate_pcc(final_model_sep, X_test, y_test)
                print(f'pcc error: {error_pcc}')
                wandb.log({"pcc": error_pcc})

                # evaluate the model correlation on training set
                error_pcc_train = evaluate_pcc(final_model_sep, X_train, y_train)
                print(f'pcc error train: {error_pcc_train}')
                wandb.log({"train_pcc": error_pcc_train})

                # evaluate the model correlation on test set based on logI and logI_prev
                error_pcc_logI = evaluate_pcc(final_model_sep, X_test, y_test, logI_test, logI_prev_test)
                print(f'pcc error logI: {error_pcc_logI}')
                wandb.log({"pcc_I": error_pcc_logI})

                # evaluate the model correlation on training set based on logI and logI_prev
                error_pcc_logI_train = evaluate_pcc(final_model_sep, X_train, y_train, logI_train, logI_prev_train)
                print(f'pcc error logI train: {error_pcc_logI_train}')
                wandb.log({"train_pcc_I": error_pcc_logI_train})

                # evaluate the model on test cme_files
                above_threshold = mae_plus_threshold
                # evaluate the model error for rare samples on test set
                error_mae_cond = evaluate_mae(
                    final_model_sep, X_test, y_test, above_threshold=above_threshold)
                print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
                wandb.log({"mae+": error_mae_cond})

                # evaluate the model error for rare samples on training set
                error_mae_cond_train = evaluate_mae(
                    final_model_sep, X_train, y_train, above_threshold=above_threshold)
                print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                wandb.log({"train_mae+": error_mae_cond_train})

                # evaluate the model correlation for rare samples on test set
                error_pcc_cond = evaluate_pcc(
                    final_model_sep, X_test, y_test, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                wandb.log({"pcc+": error_pcc_cond})

                # evaluate the model correlation for rare samples on training set
                error_pcc_cond_train = evaluate_pcc(
                    final_model_sep, X_train, y_train, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
                wandb.log({"train_pcc+": error_pcc_cond_train})

                # evaluate the model correlation for rare samples on test set based on logI and logI_prev
                error_pcc_cond_logI = evaluate_pcc(
                    final_model_sep, X_test, y_test, logI_test, logI_prev_test, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond_logI}')
                wandb.log({"pcc+_I": error_pcc_cond_logI})

                # evaluate the model correlation for rare samples on training set based on logI and logI_prev
                error_pcc_cond_logI_train = evaluate_pcc(
                    final_model_sep, X_train, y_train, logI_train, logI_prev_train, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_logI_train}')
                wandb.log({"train_pcc+_I": error_pcc_cond_logI_train})

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

                # Evaluate the model correlation with colored
                file_path = plot_repr_corr_dist(
                    final_model_sep,
                    X_train_filtered, y_train_filtered,
                    title + "_training",
                    model_type='features_reg'
                )
                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_corr_dist(
                    final_model_sep,
                    X_test_filtered, y_test_filtered,
                    title + "_test",
                    model_type='features_reg'
                )
                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Log t-SNE plot
                # Log the training t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    final_model_sep,
                    X_train_filtered, y_train_filtered, title,
                    'stage2_training',
                    model_type='features_reg',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Log the testing t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    final_model_sep,
                    X_test_filtered, y_test_filtered, title,
                    'stage2_testing',
                    model_type='features_reg',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Plot the error histograms
                filename = plot_error_hist(
                    final_model_sep,
                    X_train, y_train,
                    sample_weights=None,
                    title=title,
                    prefix='training')
                wandb.log({"training_error_hist": wandb.Image(filename)})

                # Plot the error histograms on the testing set
                filename = plot_error_hist(
                    final_model_sep,
                    X_test, y_test,
                    sample_weights=None,
                    title=title,
                    prefix='testing')
                wandb.log({"testing_error_hist": wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
