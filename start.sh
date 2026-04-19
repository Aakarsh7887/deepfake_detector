#!/usr/bin/env bash

gunicorn setting.wsgi:application --chdir App --bind 0.0.0.0:$PORT