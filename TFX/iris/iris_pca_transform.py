
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = 'variety'

def _fill_in_missing(x):
    """Replace missing valueimport numpy as np
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
    Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
    """
    default_value = '' if x.dtype == tf.string else 0
    return tf.sparse.to_dense(tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), 
                              default_value)
      
def preprocessing_fn(inputs):
    features = []
    outputs = {
        LABEL_KEY: _fill_in_missing(inputs[LABEL_KEY])
    }
    
    for feature_name, feature_tensor in inputs.items():
        if feature_name != LABEL_KEY:
            features.append(tft.scale_to_z_score( # standard scaler pre-req for PCA
                _fill_in_missing(feature_tensor)         # filling in missing values
            ))

    # concat to make feature matrix for PCA to run over
    feature_matrix = tf.concat(features, axis=1)  
    
    # get orthonormal vector matix
    orthonormal_vectors = tft.pca(feature_matrix, output_dim=2, dtype=tf.float32)
    
    # multiply matrix by feature matrix to get transformation
    pca_examples = tf.linalg.matmul(feature_matrix, orthonormal_vectors)
    
    # unstack and add to output dict
    pca_examples = tf.unstack(pca_examples, axis=1)
    outputs['Principal Component 1'] = pca_examples[0]
    outputs['Principal Component 2'] = pca_examples[1]
      
    return outputs
