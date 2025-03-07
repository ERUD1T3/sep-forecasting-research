# import os
from datetime import datetime

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import (
    plot_tsne_delta,
    plot_repr_correlation,
    plot_repr_corr_dist,
    plot_repr_corr_density,
    evaluate_pcc_repr
)
from modules.reweighting.ImportanceWeighting import exDenseReweightsD
from modules.shared.globals import *
from modules.training import cme_modeling
from modules.training.cme_modeling import pds_space_norm
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    filter_ds,
    set_seed,
    stratified_batch_dataset,
    find_optimal_epoch_by_quadratic_fit
)
from sources.attm.modules import create_attentive_model


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # left is 1


def main():
    """
    Main function to run the PDS model
    :return:
    """
    # list the devices available
    devices = tf.config.list_physical_devices('GPU')
    print(f'devices: {devices}')
    # Define the dataset options, including the sharding policy
    mb = cme_modeling.ModelBuilder()  # Model builder
    pm = TrainingPhaseManager()  # Training phase manager

    for seed in [456789]:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha, alphaV in [(0, 0)]:
                    for rho in [0]:
                        for add_slope in ADD_SLOPE:
                            # PARAMS
                            outputs_to_use = OUTPUTS_TO_USE
                            batch_size = BATCH_SIZE_PRE  # full dataset used
                            print(f'batch size : {batch_size}')

                            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                            # Construct the title
                            title = f'ATTM_PDSstratinj_bs{batch_size}_alpha{alpha:.2f}_rho{rho:.2f}_cheatQuad'

                            # Replace any other characters that are not suitable for filenames (if any)
                            title = title.replace(' ', '_').replace(':', '_')

                            # Create a unique experiment name with a timestamp
                            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                            experiment_name = f'{title}_{current_time}'
                            # Set the early stopping patience and learning rate as variables
                            set_seed(seed)
                            patience = PATIENCE_PRE
                            learning_rate = ATTM_START_LR
                            weight_decay = ATTM_WD  # higher weight decay
                            momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                            epochs = EPOCHS  # higher epochs
                            attn_hiddens = ATTN_HIDDENS
                            blocks_hiddens = BLOCKS_HIDDENS

                            attn_hiddens_str = (", ".join(map(str, attn_hiddens))).replace(', ', '_')
                            blocks_hiddens_str = (", ".join(map(str, blocks_hiddens))).replace(', ', '_')
                            pretraining = True
                            attn_dropout = ATTN_DROPOUT
                            attm_dropout = ATTM_DROPOUT
                            activation = ATTM_ACTIVATION
                            attn_norm = ATTN_NORM
                            attm_norm = ATTM_NORM
                            attn_residual = ATTN_RESIDUAL
                            attm_residual = ATTM_RESIDUAL
                            attn_skipped_layers = ATTN_SKIPPED_LAYERS
                            attm_skipped_blocks = ATTM_SKIPPED_BLOCKS

                            reduce_lr_on_plateau = ReduceLROnPlateau(
                                monitor=LR_CB_MONITOR,
                                factor=LR_CB_FACTOR,
                                patience=LR_CB_PATIENCE,
                                verbose=VERBOSE,
                                min_delta=LR_CB_MIN_DELTA,
                                min_lr=LR_CB_MIN_LR)

                            embed_dim = EMBED_DIM
                            bandwidth = BANDWIDTH
                            N = N_FILTERED  # number of samples to keep outside the threshold
                            lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                            upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                            mae_plus_threshold = MAE_PLUS_THRESHOLD

                            # Initialize wandb
                            wandb.init(project="PDS-ATTM", name=experiment_name, config={
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
                                "pds": pds,
                                "seed": seed,
                                "stage": 1,
                                "reduce_lr_on_plateau": True,
                                "attm_dropout": attm_dropout,
                                "attn_dropout": attn_dropout,
                                "activation": "LeakyReLU",
                                "attn_norm": attn_norm,
                                "attm_norm": attm_norm,
                                "optimizer": "adamw",
                                "architecture": "mlp",
                                "alpha": alpha,
                                "alphaVal": alphaV,
                                "bandwidth": bandwidth,
                                'attn_residual': attn_residual,
                                'attm_residual': attm_residual,
                                'attn_skipped_layers': attn_skipped_layers,
                                'attm_skipped_blocks': attm_skipped_blocks,
                                "embed_dim": embed_dim,
                                "ds_version": DS_VERSION,
                                "N_freq": N,
                                "lower_t": lower_threshold,
                                "upper_t": upper_threshold,
                                'mae_plus_th': mae_plus_threshold,
                                "cme_speed_threshold": cme_speed_threshold,
                                "sam_rho": rho,
                                'outputs_to_use': outputs_to_use,
                                'inj': 'strat'
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

                            # print all cme_files shapes
                            print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

                            # get the number of features
                            n_features = X_train.shape[1]
                            print(f'n_features: {n_features}')

                            # pds normalize the data
                            y_train_norm, norm_lower_t, norm_upper_t = pds_space_norm(y_train)

                            # Compute the sample weights
                            delta_train = y_train_norm[:, 0]
                            print(f'delta_train.shape: {delta_train.shape}')
                            print(f'rebalancing the training set...')
                            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                            train_weights_dict = exDenseReweightsD(
                                X_train, delta_train,
                                alpha=alpha, bw=bandwidth,
                                min_norm_weight=min_norm_weight,
                                debug=False).label_reweight_dict
                            print(f'done rebalancing the training set...')

                            X_train_filtered, y_train_filtered = filter_ds(
                                X_train, y_train,
                                low_threshold=lower_threshold,
                                high_threshold=upper_threshold,
                                N=N, seed=seed)

                            # build the test set
                            X_test, y_test = build_dataset(
                                root_dir + '/testing',
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                cme_speed_threshold=cme_speed_threshold)
                            print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                            y_test_norm, _, _ = pds_space_norm(y_test)

                            # Compute the sample weights for test
                            delta_test = y_test_norm[:, 0]
                            print(f'delta_test.shape: {delta_test.shape}')
                            print(f'rebalancing the test set...')
                            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                            test_weights_dict = exDenseReweightsD(
                                X_test, delta_test,
                                alpha=alphaV, bw=bandwidth,
                                min_norm_weight=min_norm_weight,
                                debug=False).label_reweight_dict
                            print(f'test set rebalanced.')

                            X_test_filtered, y_test_filtered = filter_ds(
                                X_test, y_test,
                                low_threshold=lower_threshold,
                                high_threshold=upper_threshold,
                                N=N, seed=seed)

                            # create the model
                            final_model_sep = create_attentive_model(
                                input_dim=n_features,
                                output_dim=0,
                                pds=pds,
                                hidden_blocks=blocks_hiddens,
                                attn_hidden_units=attn_hiddens,
                                attn_hidden_activation=activation,
                                attn_skipped_layers=attn_skipped_layers,
                                skipped_blocks=attm_skipped_blocks,
                                attn_residual=attn_residual,
                                residual=attm_residual,
                                attn_dropout=attn_dropout,
                                dropout=attm_dropout,
                                attn_norm=attn_norm,
                                norm=attm_norm,
                                embed_dim=embed_dim,
                                activation=activation,
                                sam_rho=rho
                            )

                            final_model_sep.summary()

                            # Define the EarlyStopping callback
                            early_stopping = EarlyStopping(
                                monitor=ES_CB_MONITOR,
                                patience=patience,
                                verbose=VERBOSE,
                                restore_best_weights=ES_CB_RESTORE_WEIGHTS)

                            final_model_sep.compile(
                                optimizer=AdamW(
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    beta_1=momentum_beta1
                                ),
                                loss=lambda y_true, y_pred: mb.pds_loss_vec(
                                    y_true, y_pred,
                                    phase_manager=pm,
                                    train_sample_weights=train_weights_dict,
                                    val_sample_weights=test_weights_dict,
                                )
                            )

                            train_ds, train_steps = stratified_batch_dataset(
                                X_train, y_train_norm, batch_size)
                            test_ds, test_steps = stratified_batch_dataset(
                                X_test, y_test_norm, batch_size)

                            history = final_model_sep.fit(
                                train_ds,
                                steps_per_epoch=train_steps,
                                epochs=epochs,
                                validation_data=test_ds,
                                validation_steps=test_steps,
                                batch_size=batch_size,
                                callbacks=[
                                    reduce_lr_on_plateau,
                                    early_stopping,
                                    WandbCallback(save_model=WANDB_SAVE_MODEL),
                                    IsTraining(pm)
                                ],
                                verbose=VERBOSE
                            )

                            # Save the final model
                            final_model_sep.save_weights(f"final_model_weights_{str(experiment_name)}.h5")
                            # print where the model weights are saved
                            print(f"Model weights are saved in final_model_weights_{str(experiment_name)}.h5")

                            above_threshold = norm_upper_t
                            # evaluate pcc+ on the test set
                            error_pcc_cond = evaluate_pcc_repr(
                                final_model_sep, X_test, y_test_norm, i_above_threshold=above_threshold)
                            print(f'pcc error delta i>= {above_threshold} test: {error_pcc_cond}')
                            wandb.log({"jpcc+": error_pcc_cond})

                            # evaluate pcc+ on the training set
                            error_pcc_cond_train = evaluate_pcc_repr(
                                final_model_sep, X_train, y_train_norm, i_above_threshold=above_threshold)
                            print(f'pcc error delta i>= {above_threshold} train: {error_pcc_cond_train}')
                            wandb.log({"train_jpcc+": error_pcc_cond_train})

                            # Evaluate the model correlation on the test set
                            error_pcc = evaluate_pcc_repr(final_model_sep, X_test, y_test_norm)
                            print(f'pcc error delta test: {error_pcc}')
                            wandb.log({"jpcc": error_pcc})

                            # Evaluate the model correlation on the training set
                            error_pcc_train = evaluate_pcc_repr(final_model_sep, X_train, y_train_norm)
                            print(f'pcc error delta train: {error_pcc_train}')
                            wandb.log({"train_jpcc": error_pcc_train})

                            # Evaluate the model correlation with colored
                            file_path = plot_repr_corr_dist(
                                final_model_sep,
                                X_train_filtered, y_train_filtered,
                                title + "_training"
                            )
                            wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                            print('file_path: ' + file_path)

                            file_path = plot_repr_corr_dist(
                                final_model_sep,
                                X_test_filtered, y_test_filtered,
                                title + "_test"
                            )
                            wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                            print('file_path: ' + file_path)

                            # Log t-SNE plot
                            # Log the training t-SNE plot to wandb
                            stage1_file_path = plot_tsne_delta(
                                final_model_sep,
                                X_train_filtered, y_train_filtered, title,
                                'stage1_training',
                                model_type='features',
                                save_tag=current_time, seed=seed)
                            wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                            print('stage1_file_path: ' + stage1_file_path)

                            # Log the testing t-SNE plot to wandb
                            stage1_file_path = plot_tsne_delta(
                                final_model_sep,
                                X_test_filtered, y_test_filtered, title,
                                'stage1_testing',
                                model_type='features',
                                save_tag=current_time, seed=seed)
                            wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                            print('stage1_file_path: ' + stage1_file_path)

                            # Evaluate the model correlation
                            file_path = plot_repr_correlation(
                                final_model_sep,
                                X_train_filtered, y_train_filtered,
                                title + "_training"
                            )
                            wandb.log({'representation_correlation_plot_train': wandb.Image(file_path)})
                            print('file_path: ' + file_path)

                            file_path = plot_repr_correlation(
                                final_model_sep,
                                X_test_filtered, y_test_filtered,
                                title + "_test"
                            )
                            wandb.log({'representation_correlation_plot_test': wandb.Image(file_path)})
                            print('file_path: ' + file_path)

                            # Evaluate the model correlation density
                            file_path = plot_repr_corr_density(
                                final_model_sep,
                                X_train_filtered, y_train_filtered,
                                title + "_training"
                            )
                            wandb.log({'representation_correlation_density_plot_train': wandb.Image(file_path)})
                            print('file_path: ' + file_path)

                            file_path = plot_repr_corr_density(
                                final_model_sep,
                                X_test_filtered, y_test_filtered,
                                title + "_test"
                            )
                            wandb.log({'representation_correlation_density_plot_test': wandb.Image(file_path)})
                            print('file_path: ' + file_path)

                            # Finish the wandb run
                            wandb.finish()


if __name__ == '__main__':
    main()
