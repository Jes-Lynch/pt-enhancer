import numpy as np
import scipy.misc


def save_output(lr_img, prediction, hr_img, path):
    h = prediction.shape[0]
    hr_img = do_resize(_post_process(hr_img), [h, prediction.shape[1]])
    lr_img = do_resize(_post_process(lr_img), [h, prediction.shape[1]])
    eh_img = _post_process(prediction)
    out_img = np.concatenate((lr_img, eh_img, hr_img), axis=1)
    return scipy.misc.imsave(path, out_img)


def save_image(image, path, normalize=False):
    out_img = _post_process(image)
    if normalize:
        out_img = _intensity_normalization(out_img)
    return scipy.misc.imsave(path, out_img)


def do_resize(x, shape):
    y = scipy.misc.imresize(x, shape, interp='bicubic')
    return y


def _pre_process(images):
    pre_processed = _normalize(np.asarray(images))
    pre_processed = pre_processed[:, :, np.newaxis] if len(pre_processed.shape) == 2 else pre_processed
    return pre_processed


def _intensity_normalization(image):
    threshold = 200
    image = np.where(image < threshold, (image + 40), image)
    mean = np.mean(np.where(image > threshold))
    image = np.where(image > threshold, (image - mean + 240), image)
    return image


def _post_process(images):
    post_processed = _unnormalize(images)
    return post_processed.squeeze()


def _unnormalize(image):
    return image


def _normalize(image):
    return image
