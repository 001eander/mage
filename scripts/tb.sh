#!/bin/bash

port=16007

pid=$(lsof -ti:$port 2>/dev/null)

if [ -n "$pid" ]; then
    kill $pid
    echo "Killed process $pid on port $port"
fi

echo "Starting TensorBoard on port $port..."
uv run tensorboard --logdir output_dir --port $port