'''
Script for scoring images with pre-trained models

by Jack Hessel
https://github.com/jmhessel/catrank
'''
from __future__ import print_function
import argparse
from utils import *
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Flatten
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from scipy.stats import percentileofscore as pos

def parse_args():
    parser = argparse.ArgumentParser(description = 'Multimodal Popularity Ranking')
    parser.add_argument("image", type = str,
                        help = "Path to an image file to process")
    parser.add_argument("model", type = str,
                        help = "Which model to use? [FoodPorn, cats, aww, etc.]")
    parser.add_argument("--caption", type = str, default = "",
                        help = "optional, what caption does the image have?")
    parser.add_argument("--list_mode", type = bool, default = False,
                        help = "If true, image and caption should be text files.\nimage should be the relative image path of many images, one-per-line\ntext (if given) should be the caption of each of those images, one-per-line.")
    res = parser.parse_args()
    return res

def load_lines(fp):
    with open(fp) as f:
        return [x.strip() for x in f.readlines()]

def get_image_feats(images):
    #Extract ResNet features from images
    print("Extracting image features from {} file(s)".format(len(images)))
    base_model = ResNet50(pooling='avg', include_top=False)
    gen = image_generator(images, 32)
    feats = base_model.predict_generator(gen, int(np.ceil(len(images) / 32)))
    feats = feats[:len(images),:]
    return feats
    
def main():
    args = parse_args()
    
    #should we use the multimodal model? Or just images?
    multimodal = args.caption != ""

    valid_models = set(['aww','pics','cats',
                        'FoodPorn','RedditLaqueristas','MakeupAddiction'])
    if args.model not in valid_models:
        print("{} was not in the set of valid models {}".format(args.model,
                                                                valid_models))
        quit()
    
    #Load image and caption list
    if args.list_mode:
        images = load_lines(args.image)
        if multimodal:
            captions = load_lines(args.caption)
    else:
        images = [args.image]
        if multimodal:
            captions = [args.caption]

    #Load model weights...
    print("Loading model parameters...")
    extra = 'mm' if multimodal else "uni"
    image_weight_fp = "pretrained_models/{}_{}_resnet50_weights.txt".format(args.model, extra)
    image_weights = np.array([float(x) for x in load_lines(image_weight_fp)])
    if multimodal:
        text_weight_fp = "pretrained_models/{}_{}_text_weights.txt".format(args.model, extra)
        vocab, text_weights = zip(*[x.split() for x in load_lines(text_weight_fp)])
        vocab = {w:i for i,w in enumerate(vocab)}
        text_weights = np.array([float(x) for x in text_weights]) 

    #load id2score so we can get percentiles
    id2score_fp = "pretrained_models/{}_{}_id2score.txt".format(args.model, extra)
    id2score = dict([(x.split()[0], float(x.split()[1])) for x in load_lines(id2score_fp)])
    
    #Extract image features...
    image_feats = get_image_feats(images)
    image_feats = image_feats/norm(image_feats, axis=1, ord=2)[:,None]
    if multimodal:
        print("Extracting text features...")
        captions = [preprocess_caption(x) for x in captions]
        text_feats = captions_to_matrix(captions, vocab)

    #compute the final scores as a simple dot product
    scores = image_feats.dot(image_weights)
    if multimodal:
        scores += text_feats.dot(text_weights)

    scores = scores.flatten()
    
    if multimodal:
        for img, cap, sc in zip(images, captions, scores):
            if len(cap) > 27:
                cap = cap[:27] + "..."
            if len(img) > 27:
                img = img[:27] + "..."
            print("{:<30}\t{:<30}\t{:.1f}/100".format(img, cap, pos(np.array(list(id2score.values())), sc)))
    else:
        for img, sc in zip(images, scores):
            if len(img) > 27:
                img = img[:27] + "..."
            print("{:<30}\t{:.1f}/100".format(img, pos(np.array(list(id2score.values())), sc)))

if __name__ == '__main__':
    main()
