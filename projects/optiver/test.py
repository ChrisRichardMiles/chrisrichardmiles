# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os

print(os.environ.get('NEPTUNE_API_TOKEN', ''))

if os.environ.get('NEPTUNE_API_TOKEN', ''): print('lkj')


