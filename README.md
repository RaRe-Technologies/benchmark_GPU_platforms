# benchmark-GPU-platforms
Repository to benchmark GPU providing platforms. The code/data in this repository achieves the following:
1. Trains a Bi-directional LSTM on sentiment analysis task using [twitter data](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) (more than 1.5 million tweets -- processed and contained in the `data/` directory)
2. Collects various metrics during/after training to help compare various GPU cloud platforms.
3. Pins all required library/packages versions, fixes seeds and contains a `Dockerfile` which can be used to repeat the experiments in a reproducible fashion.

### Benchmark task
A bidirectional LSTM is trained (using Keras) to perform binary categorization of tweets.
### Data
[Twitter Sentiment Analysis Dataset](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/) containing 1,578,627 classified tweets. This data is split into two files (negative and positive sentiment tweets) which can be found in the `data/` directory.
### Pull Docker Image
`docker pull manneshiva/playground:benchmark-gpu-tfsource-keras`
### Run benchmark
Make sure you create a `results/` folder before running the benchmark.

`python benchmark.py --platform aws --epochs 4 --interval 30 --data_ratio 1.0`

*usage help:* `python benchmark.py -h`

Once finished, you can find the results of the benchmark run in `results/`.
