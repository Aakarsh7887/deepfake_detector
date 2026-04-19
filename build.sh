#!/usr/bin/env bash

pip install -r requirements.txt
python App/manage.py migrate
python App/manage.py collectstatic --noinput