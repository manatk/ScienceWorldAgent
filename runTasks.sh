#!/bin/bash

for i in {0..29}
do
    python examples/sci_world_agent.py --task-num=$i --num-episodes=10 > longPrompt${i}Test.txt
done
