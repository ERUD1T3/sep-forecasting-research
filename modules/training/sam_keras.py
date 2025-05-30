from typing import List, Tuple, Union

import tensorflow as tf


class SAM:
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.

    This class implements the SAM optimization technique, which seeks to find
    parameters that lie in neighborhoods having uniformly low loss value.
    """

    def __init__(self, base_optimizer: tf.keras.optimizers.Optimizer, rho: float = 0.05, eps: float = 1e-12, norm: bool = True):
        """
        Initialize the SAM optimizer.

        Args:
            base_optimizer: The base optimizer to wrap (e.g., Adam, SGD).
            rho: Size of the neighborhood for perturbation. Default is 0.05.
            eps: Small constant for numerical stability. Default is 1e-12.
            norm: Whether to normalize the perturbation by gradient norm. Default is True.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        self.rho = rho
        self.eps = eps
        self.norm = norm
        self.base_optimizer = base_optimizer
        self.e_ws: List[tf.Tensor] = []  # Store the perturbations

    def first_step(self, gradients: List[tf.Tensor], trainable_vars: List[tf.Variable]):
        """
        Perform the first step of SAM: perturb the model parameters.

        Args:
            gradients: List of gradients for each trainable variable.
            trainable_vars: List of trainable variables of the model.
        """
        self.e_ws = []
        
        if self.norm:
            # Normalized version (original implementation)
            grad_norm = tf.linalg.global_norm(gradients)
            ew_multiplier = self.rho / (grad_norm + self.eps)
            
            for grad, var in zip(gradients, trainable_vars):
                e_w = tf.math.multiply(grad, ew_multiplier)
                var.assign_add(e_w)
                self.e_ws.append(e_w)
        else:
            # Unnormalized version
            for grad, var in zip(gradients, trainable_vars):
                e_w = tf.math.multiply(grad, self.rho)
                var.assign_add(e_w)
                self.e_ws.append(e_w)

    def second_step(self, gradients: List[tf.Tensor], trainable_variables: List[tf.Variable]):
        """
        Perform the second step of SAM: revert the perturbation and apply the update.

        Args:
            gradients: List of gradients for each trainable variable.
            trainable_variables: List of trainable variables of the model.
        """
        # Revert the perturbation
        for var, e_w in zip(trainable_variables, self.e_ws):
            var.assign_sub(e_w)

        # Apply the actual "sharpness-aware" update
        self.base_optimizer.apply_gradients(zip(gradients, trainable_variables))


def sam_train_step(
        self,
        data: Union[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]],
        rho: float = 0.05,
        eps: float = 1e-12,
        norm: bool = True
) -> dict:
    """
    Custom training step for Sharpness-Aware Minimization (SAM).

    This function overrides the default `train_step` method in Keras models
    to implement the SAM optimization technique.

    Args:
        self: The model instance.
        data: A tuple of (inputs, targets) or (inputs, targets, sample_weights).
        rho: Size of the neighborhood for perturbation. Default is 0.05.
        eps: Small constant for numerical stability. Default is 1e-12.
        norm: Whether to normalize the perturbation by gradient norm. Default is True.

    Returns:
        A dictionary containing the metric results, including the loss.
    """
    # Unpack the data
    if len(data) == 3:
        x, y, sample_weight = data
    else:
        sample_weight = None
        x, y = data

    # First forward pass and gradient computation
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(
            y, y_pred, 
            sample_weight=sample_weight, 
            regularization_losses=self.losses
        )

    # Compute gradients without perturbation
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # perturb weights
    e_ws = []
    
    if norm:
        # Normalized version (original implementation)
        grad_norm = tf.linalg.global_norm(gradients)
        ew_multiplier = rho / (grad_norm + eps)
        
        for grad, var in zip(gradients, trainable_vars):
            e_w = tf.math.multiply(grad, ew_multiplier)
            var.assign_add(e_w)  # w + e_hat(w)
            e_ws.append(e_w)
    else:
        # Unnormalized version
        for grad, var in zip(gradients, trainable_vars):
            e_w = tf.math.multiply(grad, rho)
            var.assign_add(e_w)  # w + e_hat(w)
            e_ws.append(e_w)

    # Second forward pass and gradient computation with perturbed weights
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(
            y, y_pred, 
            sample_weight=sample_weight, 
            regularization_losses=self.losses
        )

    # Compute gradients with perturbed weights
    perturbed_gradients = tape.gradient(loss, trainable_vars)

    # revert perturbation to get the original parameters
    for e_w, var in zip(e_ws, trainable_vars):
        var.assign_sub(e_w)

    # Apply the actual "sharpness-aware" update
    # using the based parameters and perturbed gradients
    self.optimizer.apply_gradients(zip(perturbed_gradients, trainable_vars))

    # Update and return metrics
    self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
    return {m.name: m.result() for m in self.metrics}


# Custom model class that uses SAM
class SAMModel(tf.keras.Model):
    """
    Custom Keras Model class that integrates Sharpness-Aware Minimization (SAM) into the training step.

    Args:
        rho: Size of the neighborhood for perturbation. Default is 0.05.
        norm: Whether to normalize the perturbation by gradient norm. Default is True.
    
    Returns:
        dict: A dictionary containing the training loss and metrics.
    """

    def __init__(self, *args, rho=0.05, norm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho = rho
        self.norm = norm

    def train_step(self, data: any) -> dict:
        """
        Overrides the default train_step method to use SAM.

        Args:
            data (tuple): A tuple containing the input data and target labels.

        Returns:
            dict: A dictionary containing the training loss and metrics.
        """
        return sam_train_step(self, data, rho=self.rho, norm=self.norm)
