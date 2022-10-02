#!/bin/bash
nvidia-docker run -it --rm \
  --net="host" \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e LD_LIBRARY_PATH='/workspace/install/lib/' \
  -v /home/automldec/github.com/rtu715/NAS-Bench-360:/NAS-Bench-360 \
  renbotu/nb360:precompute-darcyflow /bin/bash
  