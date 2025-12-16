################################################################################
#                             skforecast.stats._utils                          #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8

def check_memory_reduced(estimator, method_name: str) -> None:
    """
    Check if estimator memory has been reduced and raise informative error.
    
    Parameters
    ----------
    estimator : object
        Estimator instance to check.
    method_name : str
        Name of the method being called (for error message).
        
    Raises
    ------
    ValueError
        If estimator.memory_reduced_ is True.
    """
    if getattr(estimator, 'memory_reduced_', False):
                
        message = (
            f"Cannot call {method_name}(): model memory has been reduced via "
            f"reduce_memory() to reduce memory usage. "
            f"Refit the model to restore full functionality."
        )
        raise ValueError(message)
