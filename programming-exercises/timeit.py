#!/usr/bin/env python

import time

def timeit(fun, *args, **kwargs):
    start = time.time()
    res = fun(*args, **kwargs)
    end = time.time()
    print("%s seconds" % (end - start))
    return res
