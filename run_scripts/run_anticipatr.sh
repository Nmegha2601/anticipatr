#!/bin/bash

python src/main.py --dataset ek --root data --num_verbs 125 --num_nouns 352 --num_actions 3806 --action_repr actionset --num_queries 900 --anticipation longfuture --label_type verb --pretrained_enc_layers 3 --pretrained_path  "pretraining_expts/checkpoints/try" 


