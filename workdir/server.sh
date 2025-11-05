python3 -m sglang.launch_server \
  --model-path $1 \
  --host 0.0.0.0 \
  --port 23333 \
  --tp 8 \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --disable-cuda-graph \
  --chunked-prefill-size 1024 \
  --trust-remote-code
