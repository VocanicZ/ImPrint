"""--cpu for the benches: on a shared, loaded host wall time says more about the neighbours than the
code, so every clock the codec reads (its patience and deadlines) and the bench's own timer run in
this process's CPU seconds instead, one thread each for OpenCV and BLAS: about the seconds an idle
machine would take on one core (more, as quick_check's threads add up), and the same misses it
would see."""
import time
import types

import cv2
from threadpoolctl import threadpool_limits

import imprinted.core as ic


def use_cpu_clock():
    """Switch the codec's clock to CPU seconds on one thread; returns the clock to time with."""
    ic.time = types.SimpleNamespace(perf_counter=time.process_time)
    cv2.setNumThreads(1)
    threadpool_limits(1)
    return time.process_time
