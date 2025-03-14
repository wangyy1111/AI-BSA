import numpy as np


def dice(arr1: np.ndarray, arr2: np.ndarray, axis=(0, 1)):
    logicOr = np.sum(np.logical_and(arr1, arr2), axis)
    logicAnd = np.sum(arr1, axis) + np.sum(arr2, axis)
    return 2 * logicOr / (logicAnd + 0.000001)


def dice_with_categorical(arr1: np.ndarray, arr2: np.ndarray, axis=(0, 1)):
    arr1 = to_categorical(arr1)
    arr2 = to_categorical(arr2)
    return dice(arr1, arr2, axis)


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """
    # 将输入y向量转换为数组
    y = np.array(y, dtype='int')
    # 获取数组的行列大小
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    # y变为1维数组
    y = y.ravel()
    # 如果用户没有输入分类个数，则自行计算分类个数
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    # 生成全为0的n行num_classes列的值全为0的矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    # np.arange(n)得到每个行的位置值，y里边则是每个列的位置值
    categorical[np.arange(n), y] = 1
    # 进行reshape矫正
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
