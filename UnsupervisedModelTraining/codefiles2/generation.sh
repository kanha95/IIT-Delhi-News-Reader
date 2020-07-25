#!/bin/bash

echo "Generate text with positive sentiment "

for i in {1..10}
do

  python mLSTM_generate.py --sentiment_neuron_index=1 \
  --restore_path="./0" \
  --save_samples="pos-test" \
  --sentiment_neuron_value=2 \
  --num_chars=200
done

echo "Generate text with negative sentiment "
for i in {1..10}

do

  python mLSTM_generate.py --sentiment_neuron_index=1 \
  --restore_path="./0" \
  --save_samples="neg-test" \
  --sentiment_neuron_value=-3 \
  --num_chars=200
done
