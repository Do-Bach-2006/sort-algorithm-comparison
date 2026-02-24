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
    # Removed .tolist() to return a NumPy array for comparison
    return np.sort(array)


# -----------merge sort--------------
def merge_des(a, b):
    """
    Receive two sorted arrays a and b, merge them into one sorted array in descending order
    """
    ptra = 0
    ptrb = 0

    # Pre-allocate numpy array instead of using .append()
    res = np.empty(len(a) + len(b), dtype=a.dtype)
    i = 0

    while ptra < len(a) and ptrb < len(b):
        if a[ptra] >= b[ptrb]:
            res[i] = a[ptra]
            ptra += 1
        else:
            res[i] = b[ptrb]
            ptrb += 1
        i += 1

    # Append whatever is left over using array slicing
    if ptra < len(a):
        res[i:] = a[ptra:]
    if ptrb < len(b):
        res[i:] = b[ptrb:]

    return res


def merge_asc(a, b):
    """
    Receive two sorted arrays a and b, merge them into one sorted array in ascending order
    """
    ptra = 0
    ptrb = 0

    res = np.empty(len(a) + len(b), dtype=a.dtype)
    i = 0

    while ptra < len(a) and ptrb < len(b):
        if a[ptra] >= b[ptrb]:
            res[i] = b[ptrb]
            ptrb += 1
        else:
            res[i] = a[ptra]
            ptra += 1
        i += 1

    # Append whatever is left over
    if ptra < len(a):
        res[i:] = a[ptra:]
    if ptrb < len(b):
        res[i:] = b[ptrb:]

    return res


@benchmark
def merge_sort(array):
    """
    Merge sort function that sort the array in ascending order
    """
    res = _merge_sort(array)
    return res


def _merge_sort(array):
    """
    A merge sort helper function.
    """
    if len(array) == 0:
        return np.array([], dtype=array.dtype)

    if len(array) == 1:
        return array

    middle_index = len(array) // 2

    left = _merge_sort(array[0:middle_index])
    right = _merge_sort(array[middle_index:])

    return merge_asc(left, right)


# ------------QUick sort--------------


@benchmark
def quicksort(array):
    """
    Quick sort function that sort the array in ascending order
    """
    res = _quicksort(array)
    return res


def _quicksort(array):
    """
    Quick sort helper function
    """
    # base cases
    if len(array) == 0:
        return np.array([], dtype=array.dtype)
    if len(array) == 1:
        return array

    # chose random to minmize the worst case of quicksort
    pivot_index = random.randint(0, len(array) - 1)
    pivot = array[pivot_index]

    # Pre-allocate to respect your original for-loop logic
    left_array = np.empty(len(array), dtype=array.dtype)
    right_array = np.empty(len(array), dtype=array.dtype)
    left_ptr = 0
    right_ptr = 0

    for i in range(len(array)):
        if i == pivot_index:
            continue

        if array[i] <= pivot:
            left_array[left_ptr] = array[i]
            left_ptr += 1
        else:
            right_array[right_ptr] = array[i]
            right_ptr += 1

    # Concatenate using numpy
    return np.concatenate(
        (
            _quicksort(left_array[:left_ptr]),
            np.array([pivot], dtype=array.dtype),
            _quicksort(right_array[:right_ptr]),
        )
    )


# ------------Heap sort--------------


def reorder_min_single_node(array, root_index):
    """
    Reorder a single node in the min heap.
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
        reorder_min_single_node(array, smallest)

    return smallest


def reorder_min_heap(array, root_index):
    """
    Reorder the min heap starting from the given root index.
    """
    if root_index * 2 + 1 < len(array):
        reorder_min_heap(array, root_index * 2 + 1)
    if root_index * 2 + 2 < len(array):
        reorder_min_heap(array, root_index * 2 + 2)

    reorder_min_single_node(array, root_index)


@benchmark
def heap_sort(array):
    """
    Heap sort function that sort the array in ascending order
    """
    res = np.empty(len(array), dtype=array.dtype)
    reorder_min_heap(array, 0)

    current_size = len(array)
    i = 0

    while current_size > 0:
        # swap the min element with the very end element of the active heap
        array[0], array[current_size - 1] = array[current_size - 1], array[0]

        # "pop" it to the result array
        res[i] = array[current_size - 1]

        # decrease the active size of our heap
        current_size -= 1
        i += 1

        # shift it down using a NumPy slice (view) so len(array) behaves correctly
        if current_size > 0:
            active_heap_view = array[:current_size]
            reorder_min_single_node(active_heap_view, 0)

    return res


def experiment():
    """
    Experiment function to compare the sorting algorithms on datasets from a file.
    """
    filename = "dataset.txt"
    line_count = 0

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

                # Cast lines directly to numpy arrays
                if line_count < 5:
                    data = np.array(line.split(), dtype=np.int64)
                else:
                    data = np.array(line.split(), dtype=np.float64)

                line_count += 1
                print(f"line {line_count}")

                result1 = heap_sort(data.copy())
                result2 = merge_sort(data.copy())
                result3 = quicksort(data.copy())
                result4 = numpy_sort(data.copy())

                print("\n\n\n")

                # NumPy arrays cannot be chained with '==' for boolean evaluation.
                # We must use np.array_equal()
                is_equal = (
                    np.array_equal(result1, result2)
                    and np.array_equal(result2, result3)
                    and np.array_equal(result3, result4)
                )

                if not is_equal:
                    print("NOT EQUAL !")
                    print(len(result1))
                    print(len(result2))
                    print(len(result3))
                    continue

                print(f"result:  {is_equal}")

    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run the generator script first.")


def main():
    experiment()


if __name__ == "__main__":
    main()
