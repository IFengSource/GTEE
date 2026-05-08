#!/usr/bin/env bash

export PYTHONPATH=/root/autodl-tmp/GTEE

nnictl create --config auto_tuning/models_tuning_space_setting/Pure_GTEE/ETT_h1/Pure_GTEE_searching_config.yml
# sleep 4h
# nnictl stop --all