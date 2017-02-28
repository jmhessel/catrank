from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import string
ex = set(string.punctuation)

def load_images(image_list):
    '''
    Given a list of images, returns a numpy tensor of those images.
    '''
    images = []
    for i in image_list:
        c_img = np.expand_dims(image.img_to_array(image.load_img(i, target_size = (224, 224))), axis = 0)
        images.append(c_img)
    return preprocess_input(np.vstack(images))

def image_generator(fnames, batch_size):
    '''
    Given a list of filenames and batch size, returns image tensor batches.
    This function loops indefinitely because Keras generators are assumed to do so.
    '''
    while True:
        cfns = []
        for i, p in enumerate(fnames):
            cfns.append(p)
            if len(cfns) == batch_size:
                yield load_images(cfns)
                cfns = []
        if len(cfns) != 0:
            yield load_images(cfns)
            cfns = []

def preprocess_caption(cap):
    '''
    A minimal caption preprocessor that removes punctuation, lower-cases, etc.
    '''
    cap = ' '.join(cap.strip().lower().split())
    final_toks = []
    for tok in cap.split():
        if set(list(tok)).issubset(ex) and len(tok) > 1:
            final_toks.append(tok)
        else:
            final_toks.append(''.join([ch for ch in tok if ch not in ex]))
    return ' '.join(final_toks)


def captions_to_matrix(captions, word_to_index):
    '''
    Create an indicator matrix of unigram features for captions
    given a vocab dictionary mapping from words to indices
    '''
    mat = np.zeros([len(captions), len(word_to_index)])
    for i,c in enumerate(captions):
        for w in c.split():
            if w in word_to_index:
                mat[i, word_to_index[w]] += 1
    return mat
