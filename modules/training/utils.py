import random

import matplotlib.pyplot as plt
import numpy as np


def plot_sorted_distributions(y_train, y_val, y_test, title='Sorted Distributions'):
    """
    Plots the sorted target variable distributions for the training, validation, and test sets.

    :param:
    - y_train: Numpy array containing the training set target variable
    - y_val: Numpy array containing the validation set target variable
    - y_test: Numpy array containing the test set target variable
    - title: Title for the plot
    """
    plt.figure(figsize=(12, 6))

    # Sort and plot the target variable for each set
    plt.plot(np.sort(y_train), label='Train', marker='o')
    plt.plot(np.sort(y_val), label='Validation', marker='x')
    plt.plot(np.sort(y_test), label='Test', marker='s')

    plt.xlabel('Index')
    plt.ylabel('log_peak_intensity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def find_optimized_random_batch_size(y, threshold=np.log(10), num_trials=1000, early_exit_ratio=0.95):
    """
    Optimized function to find the minimum batch size such that, when sampled at random,
    each batch is likely to contain at least 2 rows with target values greater than a given threshold.

    Parameters:
    - y: Numpy array containing the target values
    - threshold: The target threshold, default is ln(10)
    - num_trials: Number of trials to perform for each batch size
    - early_exit_ratio: The success ratio to achieve before stopping the trials for a given batch size

    Returns:
    - Minimum batch size that fulfills the condition
    """
    # Count the number of elements greater than the threshold
    count_above_threshold = np.sum(y > threshold)

    # If there are fewer than 2 elements above the threshold, return a message
    if count_above_threshold < 2:
        raise ValueError("There are not enough samples with target values greater than the threshold.")

    # Loop through possible batch sizes, starting from 2
    for batch_size in range(2, len(y) + 1):
        print(batch_size)
        success_count = 0

        for trial in range(num_trials):
            # Sample a random batch
            random_batch = random.sample(list(y), batch_size)

            # Check if the batch contains at least 2 elements greater than the threshold
            if np.sum(np.array(random_batch) > threshold) >= 2:
                success_count += 1

            # Early exit if success ratio is achieved
            if trial >= num_trials * 0.9 and success_count / (trial + 1) >= early_exit_ratio:
                return batch_size


if __name__ == '__main__':
    # Generate random target values
    training_set_path = 'D:/College/Fall2023/sep-forecasting-research/data/electron_cme_data_split/training'

    from modules.training.ts_modeling import build_dataset

    X_train, y_train = build_dataset(training_set_path, inputs_to_use=['e0.5'], add_slope=False)
    # Find the minimum batch size
    # find the minimum batch size for the training set that satisfies the condition, using the optimized function
    min_batch_size = find_optimized_random_batch_size(y_train, threshold=4.9, num_trials=2000,
                                                      early_exit_ratio=0.99)

    print(f"Minimum batch size: {min_batch_size}")