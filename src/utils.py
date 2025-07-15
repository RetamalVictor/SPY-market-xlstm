def denormalize_value(norm_value, col: str, norm_params: dict):
    """
    Denormalizes a value (or numpy array of values) for a given feature using the provided
    normalization parameters.
    
    The normalization was performed as:
        norm(x) = 2 * (x - min) / (max - min) - 1
    so the denormalization is:
        x = ((norm(x) + 1) / 2) * (max - min) + min
    
    Args:
        norm_value: A normalized scalar, numpy array, or similar numeric type.
        col (str): The feature name corresponding to the normalization parameters.
        norm_params (dict): Normalization parameters dictionary in the format:
            {
                "inc": {"min": <value>, "max": <value>},
                "OPEN": {"min": <value>, "max": <value>},
                ...
            }
    
    Returns:
        The denormalized value(s) on the original scale.
    """
    col_min = norm_params[col]["min"]
    col_max = norm_params[col]["max"]
    original_value = ((norm_value + 1) / 2) * (col_max - col_min) + col_min
    return original_value

# Example usage:
if __name__ == "__main__":
    norm_params = {
        "inc": {
            "min": -0.023441030275643926,
            "max": 0.019607843137255054
        },
        "OPEN": {
            "min": 604.6,
            "max": 2085.2
        },
        "HIGH": {
            "min": 605.6,
            "max": 2089.2
        },
        "LOW": {
            "min": 603.1,
            "max": 2083.7
        },
        "CLOSE": {
            "min": 604.6,
            "max": 2085.2
        },
        "TIME_DIFF": {
            "min": 0.0,
            "max": 4580.0
        },
        "target": {
            "min": -0.023441030275643926,
            "max": 0.019607843137255054
        }
    }
    
    # Suppose our normalized target value is -0.01070341281592846
    norm_target = -0.01070341281592846
    original_target = denormalize_value(norm_target, "target", norm_params)
    print("Normalized target:", norm_target)
    print("Denormalized target:", original_target)
