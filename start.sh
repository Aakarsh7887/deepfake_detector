#!/usr/bin/env bash

gunicorn App.setting.wsgi:application --bind 0.0.0.0:$PORT