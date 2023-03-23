import os
import sys
import pdb
import math
import time
import datetime
import functools
import numpy as np

# general formula
def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator


def forall(func):
    """
    runs a function once per positional argument,
    with one positional argument,
    and the same set of keyword arguments.
    """
    @functools.wraps(func)
    def wrapper_forall(*args, **kwargs):
        result = list()
        for arg in args:
            result.append(func(arg, **kwargs))
        return result
    return wrapper_forall


def iterate(n):
    def inner_iterate(func):
        @functools.wraps(func)
        def wrapper_iterate(*args, **kwargs):
            result = list()
            for i in range(n):
                result.append(func(i, *args, **kwargs))
            return result
        return wrapper_iterate
    return inner_iterate


def optional(func):
    """
    runs a function only if it can run error free,
    omits it otherwise. Should be used with caution, only for
    functions that have an additional effect but whose functionality
    is not indispensable.
    """
    @functools.wraps(func)
    def wrapper_optional(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            pass
    return wrapper_optional


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f'{func.__name__!r} took: {datetime.timedelta(seconds=run_time)} [HH:MM:SS]')
        return value
    return wrapper_timer
