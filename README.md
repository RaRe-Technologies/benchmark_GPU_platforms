# benchmark-GPU-platforms
Repository to benchmark GPU providing platforms. The code/data in this repository achieves the following:
1. Trains a Bi-directional LSTM on sentiment analysis task using [twitter data](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) (more than 1.5 million tweets -- processed and contained in the `data/` directory)
2. Collects various metrics during/after training to help compare various GPU cloud platforms.
3. Pins all required library/packages versions, fixes seeds and contains a `Dockerfile` which can be used to repeat the experiments in a reproducible fashion.

TODO: complete README
