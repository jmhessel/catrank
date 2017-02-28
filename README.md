## What's in here?

This repo contains pretrained models that predict relative upvotes on
Reddit for image-only and image + text models. The subreddits these
models were trained on are [/r/pics](https://www.reddit.com/r/pics/),
[/r/aww](https://www.reddit.com/r/aww/),
[/r/cats](https://www.reddit.com/r/cats/),
[/r/FoodPorn](https://www.reddit.com/r/FoodPorn/),
[/r/MakeupAddiction](https://www.reddit.com/r/MakeupAddiction/), and
[/r/RedditLaqueristas](https://www.reddit.com/r/RedditLaqueristas/),
so if you want to know if an image would probably be upvoted within
these communities, you've come to the right place! If you want to read
more about the technical details, check out the project page and paper
[here](https://www.cs.cornell.edu/~jhessel/cats/cats.html).

## How do I score new images?

If you want to score a cute cat or dog according to the /r/aww
community, you can do...

<img src="https://github.com/jmhessel/catrank/blob/master/examples/bodhi.jpg?raw=true">

```
python score_example.py examples/bodhi.jpg aww
```

The output of this is:
```
examples/bodhi.jpg		34.8/100
```

the first column is the filename, and the second column is the score
out of 100 for the image (higher is better, so I guess my dog wouldn't
get upvoted too much!). The score is the percentile of the image among
the test split these models are trained on. If you're interested in
knowing how accurate the models here are over their test split, you
can check out the [pretrained models README]()

If you want to score a cat alongside a caption according to the
/r/cats community, you can do

```
python score_example.py examples/taz.jpg cats --caption "Please don't sit on me!"
```

which outputs
```
examples/taz.jpg		please dont sit on me		55.8/100
```

There are also some simple extentions. If you want to do score many
images/captions at once, you can use `--list_mode True`; in this case,
the image and caption arguments are assumed to be text files. The
image text file has one filename per line, and the caption text file
has one caption per line. The first line of the image file should
correspond to the first line of the caption file, and so on. As an
example, you can run

```
python score_example.py examples/example_image_list.txt --caption examples/example_caption_list.txt cats --list_mode True
```

which outputs
```
examples/bodhi.jpg		who says bulldogs cant be c...	22.1/100
examples/lizzy.jpg            	my 20 year old little girl ...	99.4/100
examples/taz.jpg              	please dont sit on me         	55.8/100
```

Unsurprisingly, the model doesn't like a dog (Bodhi) being posted in
/r/cats, though the model likes the story about an elderly cat
(Lizzy)! As an interesting experiment, you can check the effect the captions
had by running

```
python score_example.py examples/example_image_list.txt cats --list_mode True
```

and comparing to the previous output.

## I want to train my own models!

If you want to train your own models, you'll need to get the datasets
that these were trained on, which are not in the repo. They are
available for download
[here](https://www.cs.cornell.edu/~jhessel/cats/cats.html).

## Citation and Contact

If you find the models here useful, please cite our paper!

```
@inproceedings{hessel2017cats,
	title={Cats and Captions vs. Creators and the Clock: Comparing Multimodal Content to Context in Predicting Relative Popularity},
	author={Hessel, Jack and Lee, Lillian and Mimno, David},
	booktitle={Proceedings of the 26th International Conference on World Wide Web},
	year={2017},
	organization={International World Wide Web Conferences Steering Committee}
}
```

If you have any questions, you can contact jhessel@cs.cornell.edu