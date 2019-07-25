#!/bin/sh
docker build --no-cache -t res_php . ; docker run -it --rm res_php:latest > out.html
