import multiprocessing


def print_cpus():
    print(f"Your local machine has {multiprocessing.cpu_count()} CPUs.")
