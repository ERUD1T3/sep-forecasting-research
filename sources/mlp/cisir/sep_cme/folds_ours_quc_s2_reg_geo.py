# TODO: add a first stage by loading weights from repr learning and then run the following as second stage


import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_sep_corr, plot_tsne_sep
from modules.reweighting.ImportanceWeighting import MDI
from modules.shared.sep_globals import *
from modules.training.cme_modeling import ModelBuilder
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_sep_ds,
    evaluate_mae,
    evaluate_pcc,
    stratified_batch_dataset,
    set_seed,
    cmse,
    create_mlp,
    create_mlp_wUN,
    plot_error_hist,
    load_folds_sep_ds,
    plot_avsp_sep,
    filter_ds_up,
    initialize_results_dict,
    update_trial_results,
    compute_averages,
    save_results_to_csv,
)


def pdc_loss_fn(
    y_true, representations,
    phase_manager,
    mb: ModelBuilder,
    train_pdc_weight_dict=None,
    val_pdc_weight_dict=None
):
    """
    PDC loss function for representations output.
    
    Args:
        y_true: Ground truth labels
        representations: Representation vectors from model
        phase_manager: Training phase manager
        mb: ModelBuilder instance for PDC loss computation
        train_pdc_weight_dict: Training weights for PDC
        val_pdc_weight_dict: Validation weights for PDC
    """
    # Ensure y_true is 2D for PDC loss computation
    # y_true = tf.expand_dims(y_true, -1) if tf.rank(y_true) == 1 else y_true

    # print the shapes
    # print(f'y_true.shape: {y_true.shape}, representations.shape: {representations.shape}')
    
    # Compute PDC loss on representations
    pdc_loss = mb.pdc_loss_vec_geo(
        y_true, representations,
        phase_manager=phase_manager,
        train_sample_weights=train_pdc_weight_dict,
        val_sample_weights=val_pdc_weight_dict
    )
    
    return pdc_loss


def cmse_loss_fn(
    y_true, predictions,
    lambda_factor: float,
    phase_manager,
    train_mse_weight_dict=None,
    val_mse_weight_dict=None,
    train_pcc_weight_dict=None,
    val_pcc_weight_dict=None,
    normalized_weights=False,
    asym_type=None
):
    """
    CMSE loss function for predictions output.
    
    Args:
        y_true: Ground truth labels
        predictions: Predictions from model
        lambda_factor: Weight for PCC component in CMSE
        phase_manager: Training phase manager
        Other parameters: Same as cmse function
    """

    # print the shapes
    # print(f'y_true.shape: {y_true.shape}, predictions.shape: {predictions.shape}')

    # Compute the standard CMSE loss (MSE + PCC) on predictions
    mse_pcc_loss = cmse(
        y_true, predictions,
        lambda_factor=lambda_factor,
        phase_manager=phase_manager,
        train_mse_weight_dict=train_mse_weight_dict,
        val_mse_weight_dict=val_mse_weight_dict,
        train_pcc_weight_dict=train_pcc_weight_dict,
        val_pcc_weight_dict=val_pcc_weight_dict,
        normalized_weights=normalized_weights,
        asym_type=asym_type
    )
    
    return mse_pcc_loss


def main():
    """
    Testing WPCC + QUC Importance + Stratified Batching + PDC reg
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()
    # Initialize ModelBuilder for PDC loss computation
    mb = ModelBuilder()

    # get the alpha_mse, alpha_pcc, alphaV_mse, alphaV_pcc, alpha_pdc, alphaV_pdc
    alphas = [(2.4, 2.4, 1.7, 1.7, 0.1, 0.1)]
    alpha_amse = alphas[0][0]
    alpha_apcc = alphas[0][2]
    alpha_pdc = alphas[0][4]
    lambda_factor = 0.5 # LAMBDA_FACTOR  # lambda for the loss
    pdc_factor = 1 # for the importance on PDC in the loss

    # Initialize results tracking ONCE before the seed loop
    n_trials = len(TRIAL_SEEDS)
    results = initialize_results_dict(n_trials)
    results['name'] = f'geo_sepc_amse{alpha_amse:.2f}_apcc{alpha_apcc:.2f}_apdc{alpha_pdc:.2f}_pdcf{pdc_factor:.2f}_l{lambda_factor:.2f}_quc_s2'

    for seed_idx, seed in enumerate(TRIAL_SEEDS):
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc, alpha_pdc, alphaV_pdc in alphas:
            for rho in RHO:  # SAM_RHOS:
                # PARAMS
                # Construct the title
                title = f'geo_mlp_amse{alpha_mse:.2f}_apcc{alpha_pcc:.2f}_apdc{alpha_pdc:.2f}_pdcf{pdc_factor:.2f}_lambda{lambda_factor:.2f}_quc_s2'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = int(1e4) # PATIENCE  # higher patience
                learning_rate = START_LR  # starting learning rate
                asym_type = ASYM_TYPE
                lr_cb_patience = 30 # LR_CB_PATIENCE
                lr_cb_factor = LR_CB_FACTOR
                lr_cb_min_lr = LR_CB_MIN_LR
                lr_cb_min_delta = LR_CB_MIN_DELTA
                cvrg_metric = CVRG_METRIC
                cvrg_min_delta = CVRG_MIN_DELTA 
                normalized_weights = NORMALIZED_WEIGHTS
                freeze = False

                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor=LR_CB_MONITOR,
                    factor=lr_cb_factor,
                    patience=lr_cb_patience,
                    verbose=VERBOSE,
                    min_delta=lr_cb_min_delta,
                    min_lr=lr_cb_min_lr)

                weight_decay = WEIGHT_DECAY  
                batch_size = 512 # BATCH_SIZE  
                epochs = EPOCHS  
                hiddens = MLP_HIDDENS  
                # proj_hiddens = PROJ_HIDDENS
                pretraining = True
                sep_threshold = SEP_THRESHOLD

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                # proj_hiddens_str = (", ".join(map(str, proj_hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                output_dim = OUTPUT_DIM
                embed_dim = EMBED_DIM
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                skip_repr = SKIP_REPR
                skipped_layers = SKIPPED_LAYERS
                smoothing_method = SMOOTHING_METHOD
                window_size = WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = VAL_WINDOW_SIZE  # allows margin of error of 10 epochs
                n_filter = N_FILTER
                use_unit_norm = False
                model_creator = create_mlp_wUN if use_unit_norm else create_mlp

                # Initialize wandb
                wandb.init(project="2025-Papers-SEPC", name=experiment_name, config={
                    "patience": patience,
                    "learning_rate": learning_rate,
                    'min_lr': lr_cb_min_lr,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "freeze": freeze,
                    # hidden in a more readable format  (wandb does not support lists)
                    "hiddens": hiddens_str,
                    # "proj_hiddens": proj_hiddens_str,
                    "loss": 'mse_pcc_pdc',
                    "lambda": lambda_factor,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "alphaV_mse": alphaV_mse,
                    "alpha_pcc": alpha_pcc,
                    "alphaV_pcc": alphaV_pcc,
                    "alpha_pdc": alpha_pdc,
                    "alphaV_pdc": alphaV_pdc,
                    "pdc_factor": pdc_factor,
                    "bandwidth": bandwidth,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "activation": 'LeakyReLU',
                    "norm": norm,
                    'optimizer': 'adam',
                    'output_dim': output_dim,
                    'architecture': 'mlp_res_repr_pdc',
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
                    'sep_threshold': sep_threshold,
                    'n_filter': n_filter,
                    'pretraining': pretraining,
                    'use_unit_norm': use_unit_norm,
                })

                # set the root directory
                root_dir = DS_PATH
                # build the dataset
                X_train, y_train = build_sep_ds(
                    root_dir + '/sep_10mev_training.csv',
                    shuffle_data=True,
                    random_state=seed
                )
                # print the training set shapes
                print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
                # getting the reweights for training set
                delta_train = y_train
                print(f'delta_train.shape: {delta_train.shape}')
                print(f'rebalancing the training set...')
                mse_train_weights_dict = MDI(
                    X_train, delta_train,
                    alpha=alpha_mse, 
                    bandwidth=bandwidth).label_importance_map
                if alpha_pcc > 0:
                    pcc_train_weights_dict = MDI(
                        X_train, delta_train,
                        alpha=alpha_pcc, 
                        bandwidth=bandwidth).label_importance_map
                else:
                    pcc_train_weights_dict = None
                
                # Compute PDC weights if PDC is enabled
                if alpha_pdc > 0:
                    pdc_train_weights_dict = MDI(
                        X_train, delta_train,
                        alpha=alpha_pdc, 
                        bandwidth=bandwidth).label_importance_map
                else:
                    pdc_train_weights_dict = None
                print(f'training set rebalanced.')
                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                X_test, y_test = build_sep_ds(
                    root_dir + '/sep_10mev_testing.csv',
                    shuffle_data=False,
                    random_state=seed
                )
                # print the test set shapes
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                # filtering training and test sets for additional results
                X_train_filtered, y_train_filtered = filter_ds_up(
                    X_train, y_train,
                    high_threshold=sep_threshold,
                    N=n_filter, seed=seed)
                X_test_filtered, y_test_filtered = filter_ds_up(
                    X_test, y_test,
                    high_threshold=sep_threshold,
                    N=n_filter, seed=seed)

                # 4-fold cross-validation
                folds_optimal_epochs = []
                for fold_idx, (X_subtrain, y_subtrain, X_val, y_val) in enumerate(
                    load_folds_sep_ds(
                        root_dir,
                        random_state=seed,
                        shuffle=True
                    )
                ):
                    print(f'Fold: {fold_idx}')
                    # print all cme_files shapes
                    print(f'X_subtrain.shape: {X_subtrain.shape}, y_subtrain.shape: {y_subtrain.shape}')
                    print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

                    # Compute the sample weights for subtraining
                    delta_subtrain = y_subtrain
                    print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                    print(f'rebalancing the subtraining set...')
                    mse_subtrain_weights_dict = MDI(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_mse, 
                        bandwidth=bandwidth).label_importance_map
                    if alpha_pcc > 0:
                        pcc_subtrain_weights_dict = MDI(
                            X_subtrain, delta_subtrain,
                            alpha=alpha_pcc, 
                            bandwidth=bandwidth).label_importance_map
                    else:
                        pcc_subtrain_weights_dict = None
                    
                    # Compute PDC weights for subtraining
                    if alpha_pdc > 0:
                        pdc_subtrain_weights_dict = MDI(
                            X_subtrain, delta_subtrain,
                            alpha=alpha_pdc, 
                            bandwidth=bandwidth).label_importance_map
                    else:
                        pdc_subtrain_weights_dict = None
                    print(f'subtraining set rebalanced.')

                    # Compute the sample weights for validation
                    delta_val = y_val
                    print(f'delta_val.shape: {delta_val.shape}')
                    print(f'rebalancing the validation set...')
                    mse_val_weights_dict = MDI(
                        X_val, delta_val,
                        alpha=alphaV_mse, 
                        bandwidth=bandwidth).label_importance_map
                    if alphaV_pcc > 0:
                        pcc_val_weights_dict = MDI(
                            X_val, delta_val,
                            alpha=alphaV_pcc, 
                            bandwidth=bandwidth).label_importance_map
                    else:
                        pcc_val_weights_dict = None
                    
                    # Compute PDC weights for validation
                    if alphaV_pdc > 0:
                        pdc_val_weights_dict = MDI(
                            X_val, delta_val,
                            alpha=alphaV_pdc, 
                            bandwidth=bandwidth).label_importance_map
                    else:
                        pdc_val_weights_dict = None
                    print(f'validation set rebalanced.')

                    # create the model
                    model_sep = model_creator(
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
                    # Model outputs: [representations, predictions]
                    # Use a list of loss functions for each output
                    model_sep.compile(
                        optimizer=Adam(
                            learning_rate=learning_rate,
                        ),
                        loss=[
                            # Loss for representations (first output)
                            lambda y_true, y_pred_repr: pdc_loss_fn(
                                y_true, y_pred_repr,
                                phase_manager=pm,
                                mb=mb,
                                train_pdc_weight_dict=pdc_subtrain_weights_dict,
                                val_pdc_weight_dict=pdc_val_weights_dict
                            ),
                            # Loss for predictions (second output)  
                            lambda y_true, y_pred_forecast: cmse_loss_fn(
                                y_true, y_pred_forecast,
                                lambda_factor=lambda_factor,
                                phase_manager=pm,
                                train_mse_weight_dict=mse_subtrain_weights_dict,
                                val_mse_weight_dict=mse_val_weights_dict,
                                train_pcc_weight_dict=pcc_subtrain_weights_dict,
                                val_pcc_weight_dict=pcc_val_weights_dict,
                                normalized_weights=normalized_weights,
                                asym_type=asym_type
                            )
                        ],
                        loss_weights=[pdc_factor, 1.0]  # Weight PDC vs CMSE losses
                    )

                    # Step 1: Create stratified dataset for the subtraining set only
                    subtrain_ds, subtrain_steps = stratified_batch_dataset(
                        X_subtrain, y_subtrain, batch_size)

                    # Map the subtraining dataset to return {'output': y} format
                    # subtrain_ds = subtrain_ds.map(lambda x, y: (x, {'forecast_head': y}))
                    
                    # Prepare validation data without batching
                    val_data = (X_val, y_val)

                    # Train the model with the callback
                    history = model_sep.fit(
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
                    # folds_optimal_epochs.append(np.argmin(history.history[ES_CB_MONITOR]) + 1)
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

                # create the final model
                final_model_sep = model_creator(
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
                    loss=[
                        # Loss for representations (first output)
                        lambda y_true, y_pred_repr: pdc_loss_fn(
                            y_true, y_pred_repr,
                            phase_manager=pm,
                            mb=mb,
                            train_pdc_weight_dict=pdc_train_weights_dict,
                            val_pdc_weight_dict=None  # No validation weights for final training
                        ),
                        # Loss for predictions (second output)
                        lambda y_true, y_pred_forecast: cmse_loss_fn(
                            y_true, y_pred_forecast,
                            lambda_factor=lambda_factor,
                            phase_manager=pm,
                            train_mse_weight_dict=mse_train_weights_dict,
                            val_mse_weight_dict=None,  # No validation weights for final training
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            val_pcc_weight_dict=None,  # No validation weights for final training
                            normalized_weights=normalized_weights,
                            asym_type=asym_type
                        )
                    ],
                    loss_weights=[pdc_factor, 1.0]  # Weight PDC vs CMSE losses
                )  # Compile the model just like before

                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Map the training dataset to return {'output': y} format
                # train_ds = train_ds.map(lambda x, y: (x, {'forecast_head': y}))

                # Train on the full dataset
                final_model_sep.fit(
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

               

                # evaluate the model on test cme_files
                above_threshold = sep_threshold
                # evaluate the model error for rare samples on test set
                error_mae_cond = evaluate_mae(
                    final_model_sep, X_test, y_test, above_threshold=above_threshold, print_individual_errors=True)
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

                

                # Process SEP event files in the specified directory
                filename =plot_avsp_sep(
                    final_model_sep,
                    X_test, y_test,
                    title=title,
                    prefix='testing',
                    sep_threshold=sep_threshold
                )

                # Log the plot to wandb
                log_title = os.path.basename(filename)
                wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                # Process SEP event files in the specified directory
                filename = plot_avsp_sep(
                    final_model_sep,
                    X_train, y_train,
                    title=title,
                    prefix='training',
                    sep_threshold=sep_threshold
                )

                # Log the plot to wandb
                log_title = os.path.basename(filename)
                wandb.log({f'training_{log_title}': wandb.Image(filename)})

                # Evaluate the model correlation with colored points
                file_path = plot_sep_corr(
                    final_model_sep,
                    X_train_filtered, y_train_filtered,
                    title + "_training",
                    model_type='features_reg',
                    sep_threshold=sep_threshold
                )

                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_sep_corr(
                    final_model_sep,
                    X_test_filtered, y_test_filtered,
                    title + "_test",
                    model_type='features_reg',
                    sep_threshold=sep_threshold
                )
                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Log t-SNE plot
                # Log the training t-SNE plot to wandb
                stage1_file_path = plot_tsne_sep(
                    final_model_sep,
                    X_train_filtered, y_train_filtered, title,
                    'stage2_training',
                    model_type='features_reg',
                    save_tag=current_time, 
                    seed=seed,
                    sep_threshold=sep_threshold)
                wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Log the testing t-SNE plot to wandb
                stage1_file_path = plot_tsne_sep(
                    final_model_sep,
                    X_test_filtered, y_test_filtered, title,
                    'stage2_testing',
                    model_type='features_reg',
                    save_tag=current_time, 
                    seed=seed,
                    sep_threshold=sep_threshold)
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

                # Update results for this trial using the seed index + 1 for trial number
                trial_idx = seed_idx + 1
                results = update_trial_results(
                    results,
                    trial_idx,
                    mae=error_mae,
                    maep=error_mae_cond,
                    pcc=error_pcc,
                    pccp=error_pcc_cond
                )

                # Finish the wandb run
                wandb.finish()

    # After all trials are complete, compute averages and save results
    results = compute_averages(results, n_trials)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Use the title for the CSV name
    csv_filename = f"sep_cme_results_{title}.csv"
    csv_path = os.path.join(results_dir, csv_filename)
    
    # Save results to CSV
    save_results_to_csv(results, csv_path)


if __name__ == '__main__':
    main()
