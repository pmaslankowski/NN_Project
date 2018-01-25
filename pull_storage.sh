#!/bin/bash
mkdir -p sketch_rnn/datasets
mkdir -p sketch_rnn/models
mkdir -p sketch_rnn/logs
gsutil -m rsync -d -r gs://nn_data/datasets sketch_rnn/datasets
gsutil -m rsync -d -r gs://nn_data/models sketch_rnn/models
gsutil -m rsync -d -r gs://nn_data/logs sketch_rnn/logs
