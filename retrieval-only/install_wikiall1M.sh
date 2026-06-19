#!/bin/sh

curl -O https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar

mkdir -p datasets/wikiall

tar -xvf wiki_all_1M.tar -C datasets/wikiall
