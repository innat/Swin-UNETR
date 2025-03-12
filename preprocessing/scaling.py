
import tensorflow as tf
from typing import Optional


def ScaleIntensityRanged(
    a_min: float,
    a_max: float,
    b_min: Optional[float] = None,
    b_max: Optional[float] = None,
    clip: bool = False,
    dtype: tf.DType = tf.float32,
):
    def wrapper(inputs):
        image = inputs['image']
        label = inputs['label']
        scaled_image = scale_intensity_range(image, a_min, a_max, b_min, b_max, clip, dtype)
        return {'image': scaled_image, 'label': label}
        
    def scale_intensity_range(
        image,
        a_min,
        a_max,
        b_min,
        b_max,
        clip,
        dtype,
    ):
        # Cast the image to the desired output dtype
        image = tf.cast(image, dtype)
    
        # Scale the intensity values
        if b_min is not None and b_max is not None:
            # Normalize to [0, 1] first
            image = (image - a_min) / (a_max - a_min)
            # Scale to [b_min, b_max]
            image = image * (b_max - b_min) + b_min
        else:
            # If b_min or b_max is None, only normalize to [0, 1]
            image = (image - a_min) / (a_max - a_min)
    
        # Clip the values if required
        if clip and b_min is not None and b_max is not None:
            image = tf.clip_by_value(image, b_min, b_max)
    
        return image

    return wrapper