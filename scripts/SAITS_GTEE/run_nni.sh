#!/usr/bin/env bash

export PYTHONPATH=/root/autodl-tmp/GTEE

nnictl create --config auto_tuning/models_tuning_space_setting/SAITS_GTEE/ETT_h1/SAITS_GTEE_searching_config.yml
# sleep 4h
# nnictl stop --all