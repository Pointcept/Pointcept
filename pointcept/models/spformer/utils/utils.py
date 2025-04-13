"""
# This file includes code adapted from SPFormer:
# https://github.com/sunjiahao1999/SPFormer
# Original author: Sun Jiahao (@sunjiahao1999)
"""

import functools


def cuda_cast(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if hasattr(x, "cuda"):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(v, "cuda"):
                v = v.cuda()
            elif isinstance(v, list) and hasattr(v[0], "cuda"):
                try:
                    v = [x.cuda() if hasattr(x, "cuda") else x for x in v]
                except:
                    print(x, v, "FAILURE")
                    # import pdb
                    # pdb.set_trace()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper
