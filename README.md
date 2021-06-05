# chrisrichardmiles
> Helpful tools for machine learning and coding.


This file will become your README and also the index of your documentation.

## Install

`pip install chrisrichardmiles`

## How to use

Simply import functions youd like to use and use them.

```python
from chrisrichardmiles.core import dict_to_str
```

```python
dict_to_str(d={
    'name': 'f1',
    'foo': 3,
    'bar': 'zoo',
    'bool': True,
    'bool2': False,
    'float': 3.45})
```




    'bar::zoo::bool::True::bool2::False::float::3.45::foo::3::name::f1'


