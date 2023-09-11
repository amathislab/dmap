#!/bin/bash

echo "Running evaluation..."

steps=(10 15 20 50 100 500 1000 2000 5000 10000 50000 100000)
# ${steps[@]};
for step in {36..49..1}
do
    echo "Number of steps: $step"
    python main_evaluation.py --steps $step
done