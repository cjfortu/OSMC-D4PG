'''
## l2_projection ##
# Taken from: https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py
# Projects the target distribution onto the support of the original network [Vmin, Vmax]
'''

import tensorflow as tf

@tf.function
def _l2_project(z_p, p, z_q):
    """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.
    The supports z_p and z_q are specified as tensors of distinct atoms (given
    in ascending order).
    Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
    support z_q, in particular Kq need not be equal to Kp.
    Args:
      z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
      p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
      z_q: Tensor holding support to project onto, shape `[Kq]`.
    Returns:
      Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    # Broadcasting of tensors is used extensively in the code below. To avoid
    # accidental broadcasting along unintended dimensions, tensors are defensively
    # reshaped to have equal number of dimensions (3) throughout and intended
    # shapes are indicated alongside tensor definitions. To reduce verbosity,
    # extra dimensions of size 1 are inserted by indexing with `None` instead of
    # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
    # `[k, l]' to one of shape `[k, 1, l]`).

    # Extract vmin and vmax and construct helper tensors from z_q
    # z_p = tf.cast(z_p, dtype=tf.double)
    p = tf.cast(p, dtype=tf.float32)
    # z_q = tf.cast(z_q, dtype=tf.double)
    # print('z_p: {} {} {}'.format(type(z_p), z_p.dtype, z_p))
    # print('p: {} {}'.format(type(p), p))
    # print('z_q: {} {}'.format(type(z_q), z_q))
    z_q = tf.cast(z_q, dtype=tf.float32)
    # p = tf.cast(p, dtype=tf.float64)
    # z_p = tf.cast(z_p, dtype=tf.float32)
    vmin, vmax = z_q[0], z_q[-1]
    # print('vmin: {}\nvmax: {}\nz_q: {}\n'.format(vmin[None].shape, vmin[None].shape, z_q.shape))
    d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
    d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
    # print('d_pos: {}\nd_neg: {}'.format(d_pos, d_neg))
    # Clip z_p to be in new support range (vmin, vmax).
    # print('z_p: {} {} {}'.format(type(z_p), z_p.dtype, z_p))
    # print('vmin: {} {} {}'.format(type(vmin), vmin.dtype, vmin))
    # print('vmax: {} {} {}'.format(type(vmax), vmax.dtype, vmax))
    z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp

    # Get the distance between atom values in support.
    d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
    d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
    z_q = z_q[None, :, None]  # 1 x Kq x 1

    # Ensure that we do not divide by zero, in case of atoms of identical value.
    d_neg = tf.compat.v1.where(d_neg > 0, 1./d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
    d_pos = tf.compat.v1.where(d_pos > 0, 1./d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1

    # print('z_p: {}\nz_q: {}\n'.format(z_p.shape, z_q.shape))
    delta_qp = z_p - z_q   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
    d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp
    # d_sign = delta_qp >= 0

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
    # Shape  B x Kq x Kp.
    delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
    p = p[:, None, :]  # B x 1 x Kp.
    return tf.reduce_sum(input_tensor=tf.clip_by_value(1. - delta_hat, 0., 1.) * p, axis=2)