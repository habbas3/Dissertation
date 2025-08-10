
import numpy as np
import random
from scipy.signal import resample
import torch



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        """Ensure sequences have shape (channels, length) and float32 dtype.

        Earlier versions returned a ``torch.Tensor`` directly, which broke
        subsequent numpy-based augmentations (e.g. ``Retype``) that expect
        ``ndarray`` inputs.  We now keep the data as a numpy array so that all
        augmentation steps operate on a consistent type, deferring conversion to
        ``torch.Tensor`` until the dataset's ``__getitem__``.
        """

        # Convert to numpy array and drop any singleton dimensions introduced
        # by certain loaders (e.g. ``(length, 1)`` windows).
        seq = np.array(seq)

        if seq.ndim == 3:
            # squeeze trailing dimension, keeping (N, L)
            if seq.shape[-1] == 1:
                seq = np.squeeze(seq, axis=-1)
            else:
                # For unexpected 3D shapes, flatten first dimension(s)
                seq = seq.reshape(seq.shape[0], -1)

        # Handle different shapes dynamically
        if seq.ndim == 1:
            # (N,) → (1, N)
            seq = seq[np.newaxis, :]
        elif seq.ndim == 2 and seq.shape[0] > seq.shape[1]:
            seq = seq.T  # Transpose if needed

        return seq.astype(np.float32)



class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


# class Scale(object):
#     def __init__(self, sigma=0.01):
#         self.sigma = sigma
#
#     def __call__(self, seq):
#         scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
#         scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
#         return seq*scale_matrix


class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomTimeShift(object):
    """Randomly shift the sequence along the temporal axis."""

    def __init__(self, shift_ratio=0.2):
        self.shift_ratio = shift_ratio

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        length = seq.shape[1]
        max_shift = int(length * self.shift_ratio)
        if max_shift < 1:
            return seq
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            padded = np.pad(seq, ((0, 0), (shift, 0)), mode='edge')
            return padded[:, :-shift]
        elif shift < 0:
            shift = -shift
            padded = np.pad(seq, ((0, 0), (0, shift)), mode='edge')
            return padded[:, shift:]
        return seq


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","-1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq =(seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq

# class Scale(object):
#     def __init__(self, factor=1.0):
#         self.factor = factor
#
#     def __call__(self, seq):
#         seq = seq*self.factor
#         return seq
