from torch import nn
from torchinfo import summary

def weights_init(m):
    """
    This function uses the technique discussed in the DC Gan's paper to initialize weights with a specific mean and std.
    Args:
        m: The model to initialize the weights for

    Returns:
        None
    """
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_model_summary(summary_model, input_shape) -> summary:
    """
    Find the model summary (parameters, input shape, output shape, trainable parameters etc.)
    Args:
        summary_model (nn.Module): The model for which we want to print the summary details
        input_shape (shape): The input shape that the model takes

    Returns:
        A summary object with the model details in it
    """
    return summary(
        summary_model,
        input_size=input_shape,
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"],
    )
