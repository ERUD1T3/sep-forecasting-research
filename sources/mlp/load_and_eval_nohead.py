import os
from datetime import datetime

import numpy as np
import wandb

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
from modules.shared.globals import *
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    create_mlp,
    plot_error_hist,
    filter_ds,
    remove_forecast_head
)

# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

weight_path = BETTER_REPR_FOR_COMBINER

def main():
    """
    Main function to load and evaluate the E-MLP model
    
    :return:
    """

    for seed in [456789]:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS:
            for rho in RHO:  # SAM_RHOS:

                inputs_to_use = INPUTS_TO_USE[0]
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]
                add_slope = ADD_SLOPE[0]
                # PARAMS
                outputs_to_use = OUTPUTS_TO_USE
                title = f'mlp2_amse{alpha_mse:.2f}_better_repr_for_combiner_wd'
                title = title.replace(' ', '_').replace(':', '_')
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'

                hiddens = MLP_HIDDENS
                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                embed_dim = EMBED_DIM
                output_dim = len(outputs_to_use)
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                skip_repr = SKIP_REPR
                skipped_layers = SKIPPED_LAYERS
                N = N_FILTERED
                lower_threshold = LOWER_THRESHOLD
                upper_threshold = UPPER_THRESHOLD
                mae_plus_threshold = MAE_PLUS_THRESHOLD

                # Initialize wandb
                wandb.init(project="Jan-Report-Evals", name=experiment_name + "_eval", config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "hiddens": hiddens_str,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "activation": 'LeakyReLU',
                    "norm": norm,
                    'output_dim': output_dim,
                    'architecture': 'mlp_res_repr',
                    'skip_repr': skip_repr,
                    'cme_speed_threshold': cme_speed_threshold,
                    'ds_version': DS_VERSION,
                    'mae_plus_th': mae_plus_threshold,
                    'sam_rho': rho,
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

                print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                X_test, y_test, logI_test, logI_prev_test = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)
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

                # Create and load the model
                model = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    embed_dim=embed_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho,
                    weight_decay=0.1
                )

                # Load the weights
                model.load_weights(weight_path)
                print(f"Loaded weights from {weight_path}")
                
                # Remove the forecast head
                model_no_head = remove_forecast_head(model)
                print("Removed forecast head from the model")

                # evaluate the model error on test set
                error_mae = evaluate_mae(model, X_test, y_test)
                print(f'mae error: {error_mae}')
                wandb.log({"mae": error_mae})

                # evaluate the model error on training set
                error_mae_train = evaluate_mae(model, X_train, y_train)
                print(f'mae error train: {error_mae_train}')
                wandb.log({"train_mae": error_mae_train})

                # evaluate the model correlation on test set
                error_pcc = evaluate_pcc(model, X_test, y_test)
                print(f'pcc error: {error_pcc}')
                wandb.log({"pcc": error_pcc})

                # evaluate the model correlation on training set
                error_pcc_train = evaluate_pcc(model, X_train, y_train)
                print(f'pcc error train: {error_pcc_train}')
                wandb.log({"train_pcc": error_pcc_train})

                # evaluate the model correlation on test set based on logI and logI_prev
                error_pcc_logI = evaluate_pcc(model, X_test, y_test, logI_test, logI_prev_test)
                print(f'pcc error logI: {error_pcc_logI}')
                wandb.log({"pcc_I": error_pcc_logI})

                # evaluate the model correlation on training set based on logI and logI_prev
                error_pcc_logI_train = evaluate_pcc(model, X_train, y_train, logI_train, logI_prev_train)
                print(f'pcc error logI train: {error_pcc_logI_train}')
                wandb.log({"train_pcc_I": error_pcc_logI_train})

                # evaluate the model on test cme_files
                above_threshold = mae_plus_threshold
                # evaluate the model error for rare samples on test set
                error_mae_cond = evaluate_mae(
                    model, X_test, y_test, above_threshold=above_threshold)
                print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
                wandb.log({"mae+": error_mae_cond})

                # evaluate the model error for rare samples on training set
                error_mae_cond_train = evaluate_mae(
                    model, X_train, y_train, above_threshold=above_threshold)
                print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                wandb.log({"train_mae+": error_mae_cond_train})

                # evaluate the model correlation for rare samples on test set
                error_pcc_cond = evaluate_pcc(
                    model, X_test, y_test, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                wandb.log({"pcc+": error_pcc_cond})

                # evaluate the model correlation for rare samples on training set
                error_pcc_cond_train = evaluate_pcc(
                    model, X_train, y_train, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
                wandb.log({"train_pcc+": error_pcc_cond_train})

                # evaluate the model correlation for rare samples on test set based on logI and logI_prev
                error_pcc_cond_logI = evaluate_pcc(
                    model, X_test, y_test, logI_test, logI_prev_test, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond_logI}')
                wandb.log({"pcc+_I": error_pcc_cond_logI})

                # evaluate the model correlation for rare samples on training set based on logI and logI_prev
                error_pcc_cond_logI_train = evaluate_pcc(
                    model, X_train, y_train, logI_train, logI_prev_train, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_logI_train}')
                wandb.log({"train_pcc+_I": error_pcc_cond_logI_train})

                # Process SEP event files in the specified directory
                test_directory = root_dir + '/testing'
                filenames = process_sep_events(
                    test_directory,
                    model,
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
                    model,
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
                    model_no_head,
                    X_train_filtered, y_train_filtered,
                    title + "_training",
                    model_type='features'
                )
                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_corr_dist(
                    model_no_head,
                    X_test_filtered, y_test_filtered,
                    title + "_test",
                    model_type='features'
                )
                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Log t-SNE plot
                stage1_file_path = plot_tsne_delta(
                    model_no_head,
                    X_train_filtered, y_train_filtered, title,
                    'stage2_training',
                    model_type='features',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                stage1_file_path = plot_tsne_delta(
                    model_no_head,
                    X_test_filtered, y_test_filtered, title,
                    'stage2_testing',
                    model_type='features',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # # Plot the error histograms
                # filename = plot_error_hist(
                #     model,
                #     X_train, y_train,
                #     sample_weights=None,
                #     title=title,
                #     prefix='training')
                # wandb.log({"training_error_hist": wandb.Image(filename)})

                # filename = plot_error_hist(
                #     model,
                #     X_test, y_test,
                #     sample_weights=None,
                #     title=title,
                #     prefix='testing')
                # wandb.log({"testing_error_hist": wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
