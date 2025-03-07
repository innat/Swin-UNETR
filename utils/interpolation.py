import tensorflow as tf

def linear_interpolation(volume, target_depth):
    # Get the original depth
    original_depth = tf.shape(volume)[0]
    
    # Generate floating-point indices for the target depth
    indices = tf.linspace(0.0, tf.cast(original_depth - 1, tf.float32), target_depth)
    
    # Split indices into integer and fractional parts
    lower_indices = tf.cast(tf.floor(indices), tf.int32)
    upper_indices = tf.minimum(lower_indices + 1, original_depth - 1)
    alpha = indices - tf.cast(lower_indices, tf.float32)  # Fractional part
    
    # Gather the lower and upper slices
    lower_slices = tf.cast(tf.gather(volume, lower_indices, axis=0), tf.float32)
    upper_slices = tf.cast(tf.gather(volume, upper_indices, axis=0), tf.float32)
    
    # Perform linear interpolation
    interpolated_volume = (1 - alpha[:, None, None]) * lower_slices + alpha[:, None, None] * upper_slices
    
    return interpolated_volume

def cubic_interpolation(volume, target_depth):
    # Get the original depth
    original_depth = tf.shape(volume)[0]
    
    # Generate floating-point indices for the target depth
    indices = tf.linspace(0.0, tf.cast(original_depth - 1, tf.float32), target_depth)
    
    # Split indices into integer and fractional parts
    lower_indices = tf.cast(tf.floor(indices), tf.int32)
    alpha = indices - tf.cast(lower_indices, tf.float32)  # Fractional part
    
    # Gather the four neighboring slices
    indices_0 = tf.maximum(lower_indices - 1, 0)
    indices_1 = lower_indices
    indices_2 = tf.minimum(lower_indices + 1, original_depth - 1)
    indices_3 = tf.minimum(lower_indices + 2, original_depth - 1)
    
    slices_0 = tf.gather(volume, indices_0, axis=0)
    slices_1 = tf.gather(volume, indices_1, axis=0)
    slices_2 = tf.gather(volume, indices_2, axis=0)
    slices_3 = tf.gather(volume, indices_3, axis=0)
    
    # Cubic interpolation coefficients
    alpha_sq = alpha ** 2
    alpha_cu = alpha ** 3
    w0 = -0.5 * alpha_cu + 1.0 * alpha_sq - 0.5 * alpha
    w1 = 1.5 * alpha_cu - 2.5 * alpha_sq + 1.0
    w2 = -1.5 * alpha_cu + 2.0 * alpha_sq + 0.5 * alpha
    w3 = 0.5 * alpha_cu - 0.5 * alpha_sq
    
    # Perform cubic interpolation
    interpolated_volume = (
        w0[:, None, None] * tf.cast(slices_0, 'float32') +
        w1[:, None, None] * tf.cast(slices_1, 'float32') +
        w2[:, None, None] * tf.cast(slices_2, 'float32') +
        w3[:, None, None] * tf.cast(slices_3, 'float32')
    )
    
    return interpolated_volume


def nearest_interpolation(volume, target_depth):
    # Generate floating-point indices for the target depth
    depth_indices = tf.linspace(0.0, tf.cast(tf.shape(volume)[0] - 1, tf.float32), target_depth)
    
    # Round the indices to the nearest integer (nearest-neighbor interpolation)
    depth_indices = tf.cast(depth_indices, tf.int32)
    
    # Gather slices from the original volume using the rounded indices
    resized_volume = tf.gather(volume, depth_indices, axis=0)
    
    return resized_volume

def depth_interpolation(volume, target_depth, method='linear'):

    SUPPORTED_METHOD = ('linear', 'nearest', 'cubic')
    
    if method not in SUPPORTED_METHOD:
        raise ValuerError(
            f'Support interplation methods are {SUPPORTED_METHOD} '
            f'But got {method}'
        )

    methods = {
        SUPPORTED_METHOD[0]: linear_interpolation,
        SUPPORTED_METHOD[1]: nearest_interpolation,
        SUPPORTED_METHOD[2]: cubic_interpolation,
    }

    return methods.get(method)(volume, target_depth)
