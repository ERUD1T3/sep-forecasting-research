import os
from datetime import datetime

import numpy as np
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_sep, plot_sep_corr
from modules.reweighting.ImportanceWeighting import ReciprocalImportance
from modules.shared.sep_globals import *
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
    plot_error_hist,
    plot_avsp_sep,
    filter_ds_up
)



def main():
    """
    Testing WPCC + Reciprocal Importance + Stratified Batching
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    for seed in [456789]:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in [(0.85, 0.85, 0.0, 0.0)]:
            for rho in RHO:  # SAM_RHOS:
                # PARAMS
                lambda_factor = 0.2 # LAMBDA_FACTOR  # lambda for the loss
                # Construct the title
                title = f'mlp_amse{alpha_mse:.2f}_apcc{alpha_pcc:.2f}_cheat'
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
                sep_threshold = SEP_THRESHOLD

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
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

                # Initialize wandb
                wandb.init(project="NeurIPS-2025-Paper-SEPds", name=experiment_name, config={
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
                    'n_filter': n_filter
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

                X_test, y_test = build_sep_ds(
                    root_dir + '/sep_10mev_testing.csv',
                    shuffle_data=False,
                    random_state=seed
                )
                # print the test set shapes
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                # rebalancing the test set for validation
                delta_test = y_test
                print(f'delta_test.shape: {delta_test.shape}')
                print(f'rebalancing the test set...')
                mse_test_weights_dict = ReciprocalImportance(
                    X_test, delta_test,
                    alpha=alphaV_mse, 
                    bandwidth=bandwidth).label_importance_map
                pcc_test_weights_dict = ReciprocalImportance(
                    X_test, delta_test,
                    alpha=alphaV_pcc, 
                    bandwidth=bandwidth).label_importance_map
                print(f'test set rebalanced.')

                # filtering training and test sets for additional results
                X_train_filtered, y_train_filtered = filter_ds_up(
                    X_train, y_train,
                    high_threshold=sep_threshold,
                    N=n_filter, seed=seed)
                X_test_filtered, y_test_filtered = filter_ds_up(
                    X_test, y_test,
                    high_threshold=sep_threshold,
                    N=n_filter, seed=seed)

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
                            train_mse_weight_dict=mse_train_weights_dict,
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            val_mse_weight_dict=mse_test_weights_dict,
                            val_pcc_weight_dict=pcc_test_weights_dict,
                            normalized_weights=normalized_weights,
                            asym_type=asym_type
                        )
                    }
                )

                # Create stratified dataset for training
                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Map the training dataset to return {'output': y} format
                train_ds = train_ds.map(lambda x, y: (x, {'forecast_head': y}))
                
                # Prepare validation data without batching
                val_data = (X_test, {'forecast_head': y_test})

                # Train the model with the callback
                history = model_sep.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
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

                # Use the function to find the optimal epoch
                optimal_epoch = find_optimal_epoch_by_smoothing(
                    history.history[ES_CB_MONITOR],
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')
                print(f'optimal_epoch: {optimal_epoch}')
                wandb.log({'optimal_epoch': optimal_epoch})

                # Create a new model for final training
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
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=optimal_epoch,
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

                # Evaluate the model correlation with colored
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

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
