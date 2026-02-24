import time
from functools import wraps
import sys
import random
import numpy as np


sys.setrecursionlimit(
    2000000
)  # Set higher recursion depth for the big dataset ( no optimize, but at least, get the job done)


# wrapper function use for time calculating
def benchmark(func):
    """
    A decorator that calculates the time a function takes to execute and print it out in console

    args:
        func is a function name


    return the result of the sort function

    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = end_time - start_time
        print(f"Execution of {func.__name__} took {duration:.4f} seconds.")
        return result

    return wrapper


# ------------Numpy sort--------------
@benchmark
def numpy_sort(array):
    return np.sort(array).tolist()


# -----------merge sort--------------
def merge_des(a, b):
    """
    Receive two sorted list a and b, merge them into one sorted list in descending order
    args:
        a is a list
        b is a list

    return:
        res a list combine a and b in the descending order from left to right
    """
    ptra = 0
    ptrb = 0

    res = []

    while ptra < len(a) and ptrb < len(b):
        if a[ptra] >= b[ptrb]:
            res.append(a[ptra])
            ptra += 1
        else:
            res.append(b[ptrb])
            ptrb += 1
    # Append whatever is left over
    res.extend(a[ptra:])
    res.extend(b[ptrb:])

    return res


def merge_asc(a, b):
    """
    Receive two sorted list a and b, merge them into one sorted list in ascending order
    args:
        a is a list
        b is a list

    return:
        res a list combine a and b in the Ascending  order from left to right
    """
    ptra = 0
    ptrb = 0

    res = []

    while ptra < len(a) and ptrb < len(b):
        if a[ptra] >= b[ptrb]:
            res.append(b[ptrb])
            ptrb += 1
        else:
            res.append(a[ptra])
            ptra += 1
    # Append whatever is left over
    res.extend(a[ptra:])
    res.extend(b[ptrb:])
    return res


# this merege sort is sort the element in ascending order


@benchmark
def merge_sort(array):
    """
    Merge sort function that sort the array in ascending order
    args:
        array is a list of elements
    return:
        res is a sorted list of elements in ascending order
    """
    # we have to call the helper to hide away from the benchmark decorator
    res = _merge_sort(array)
    return res


def _merge_sort(array):
    """
    A merge sort helper function.
    args:
        array is a list of elements
    return:
        res is a sorted list of elements in ascending order
    """

    if len(array) == 0:
        return []

    if len(array) == 1:
        return array

    middle_index = len(array) // 2

    # we sort the half left
    left = _merge_sort(array[0:middle_index])
    right = _merge_sort(array[middle_index:])

    return merge_asc(left, right)


# ------------QUick sort--------------


# this quicksort sort elemen in ascending order
@benchmark
def quicksort(array):
    """
    Quick sort function that sort the array in ascending order
    args:
        array is a list of elements
    return:
        res is a sorted list of elements in ascending order
    """
    # once again, call the helper to hide away from the benchmark decorator
    res = _quicksort(array)
    return res


def _quicksort(array):
    """
    Quick sort helper function
    args:
        array is a list of elements
    return:
        res is a sorted list of elements in ascending order
    """

    # base cases
    if len(array) == 0:
        return []
    if len(array) == 1:
        return array

    # chose random to minmize the worst case of quicksort
    pivot_index = random.randint(0, len(array) - 1)
    pivot = array[pivot_index]
    left_array = []
    right_array = []

    for i in range(len(array)):
        if i == pivot_index:
            continue

        if array[i] <= pivot:
            left_array.append(array[i])
        else:
            right_array.append(array[i])

    return _quicksort(left_array) + [pivot] + _quicksort(right_array)


# ------------Heap sort--------------

# this heap sort sort the array in ascending order


def reorder_min_single_node(array, root_index):
    """
    Reorder a single node in the min heap. If the node is larger than any of its children, swap it with the smallest child and continue reordering downwards.
    args:
        array is a list of elements
        root_index is the index of the node to reorder
    return:
        smallest is the index where the element moved to

    """
    size = len(array)
    smallest = root_index
    left = 2 * root_index + 1
    right = 2 * root_index + 2

    if left < size and array[left] < array[smallest]:
        smallest = left
    if right < size and array[right] < array[smallest]:
        smallest = right

    if smallest != root_index:
        array[root_index], array[smallest] = array[smallest], array[root_index]

        reorder_min_single_node(array, smallest)  # make sure that

    return smallest  # Tells us where the element moved to


def reorder_min_heap(array, root_index):
    """
    Reorder the min heap starting from the given root index.
    args:
        array is a list of elements
        root_index is the index of the root node to start reordering from
    return:
        None
    """
    if root_index * 2 + 1 < len(array):  # continue reorder the left
        reorder_min_heap(array, root_index * 2 + 1)
    if root_index * 2 + 2 < len(array):  # continue reorder the right
        reorder_min_heap(array, root_index * 2 + 2)

    # since left and right are already in min heap, we reorder this
    reorder_min_single_node(array, root_index)


@benchmark
def heap_sort(array):
    """
    Heap sort function that sort the array in ascending order
    args:
        array is a list of elements
    return:
        res is a sorted list of elements in ascending order
    """
    res = []

    # we construct the heap
    reorder_min_heap(array, 0)

    while len(array) > 0:
        # swap the min element with the very end element
        array[0], array[-1] = array[-1], array[0]

        # pop it to the result array
        res.append(array.pop())

        # shift it down
        reorder_min_single_node(array, 0)

    return res


def experiment():
    """
    Experiment function to compare the sorting algorithms on datasets from a file. This function is designed
    to read datasets from 'dataset.txt', apply multiple sorting algorithms, and verify that they all produce the same sorted output.
    args:
        None
    return:
        None
    """

    filename = "dataset.txt"

    line_count = 0

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                # We convert from lines to array.
                # map() is efficient for converting strings to ints and already split line to list

                data = []

                if line_count < 5:
                    data = list(map(int, line.split()))
                else:
                    data = list(map(float, line.split()))

                line_count += 1

                print(f"line {line_count}")

                result1 = heap_sort(data.copy())
                result2 = merge_sort(data.copy())
                result3 = quicksort(data.copy())
                result4 = numpy_sort(data.copy())

                print("\n\n\n")

                is_equal = result1 == result2 == result3 == result4
                if not is_equal:
                    print("NOT EQUAL !")
                    print(len(result1))
                    print(len(result2))
                    print(len(result3))
                    continue

                print(
                    f"result:  {is_equal}",
                )

    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run the generator script first.")


def main():
    # array = [random.randint(0, 100) for i in range(15)]

    # array = [9, 8, 7, 6, 5, 3, 2, 1, 10, 2, 4]

    # print(quicksort((array.copy())))
    # print(merge_sort(array.copy()))
    # print(heap_sort(array.copy()))
    #

    # NOTE: it is worth noting that these implementation of sorting algorithms isn't the optimized version !
    # you can perform in place swap, not creating new result array and getting rid of extra space complexity...
    # But for simple implementation. This code works, and it reflect the core method of these sort.

    experiment()


if __name__ == "__main__":
    main()
