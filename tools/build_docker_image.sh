#!/usr/bin/env bash
set -euo pipefail

# Build docker image with project name as image name
docker build -t dafne .
