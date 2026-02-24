import time
from functools import wraps
import sys
import random
import numpy as np
from numba import njit  # Import the JIT compiler


sys.setrecursionlimit(2000000)  # Set higher recursion depth for the big dataset


# wrapper function use for time calculating
def benchmark(func):
    """
    A decorator that calculates the time a function takes to execute and print it out in console
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        duration = (end_time - start_time) * 1000
        print(f"Execution of {func.__name__} took {duration:.1f} miliseconds.")
        return result

    return wrapper


# ------------Numpy sort--------------
@benchmark
def numpy_sort(array):
    return np.sort(array)


# -----------merge sort--------------


@njit
def merge_asc(array, temp, left, mid, right):
    """
    In-place merge using pointers and a single temporary array to avoid memory allocation overhead.
    """
    ptra = left
    ptrb = mid + 1
    i = left

    while ptra <= mid and ptrb <= right:
        if array[ptra] <= array[ptrb]:
            temp[i] = array[ptra]
            ptra += 1
        else:
            temp[i] = array[ptrb]
            ptrb += 1
        i += 1

    # Append whatever is left over
    while ptra <= mid:
        temp[i] = array[ptra]
        ptra += 1
        i += 1

    while ptrb <= right:
        temp[i] = array[ptrb]
        ptrb += 1
        i += 1

    # Copy the sorted temp chunk back into the main array
    for j in range(left, right + 1):
        array[j] = temp[j]


@njit
def _merge_sort(array, temp, left, right):
    """
    A merge sort helper function using pointers.
    """
    if left < right:
        middle_index = (left + right) // 2
        _merge_sort(array, temp, left, middle_index)
        _merge_sort(array, temp, middle_index + 1, right)
        merge_asc(array, temp, left, middle_index, right)


@benchmark
def merge_sort(array):
    """
    Merge sort function that sort the array in ascending order
    """
    # Pre-allocate one single temp array for the entire process to share
    temp = np.empty_like(array)
    _merge_sort(array, temp, 0, len(array) - 1)
    return array


# ------------QUick sort--------------


@njit
def _quicksort(array, low, high):
    """
    Quick sort helper function using strictly in-place swapping with pointers.
    """
    if low < high:
        # chose random to minmize the worst case of quicksort
        pivot_index = np.random.randint(low, high + 1)

        # Swap pivot to the end temporarily
        array[pivot_index], array[high] = array[high], array[pivot_index]
        pivot = array[high]

        left_ptr = low - 1

        for i in range(low, high):
            if array[i] <= pivot:
                left_ptr += 1
                # In-place swap
                array[left_ptr], array[i] = array[i], array[left_ptr]

        # Put pivot in its final correct place
        array[left_ptr + 1], array[high] = array[high], array[left_ptr + 1]
        final_pivot_index = left_ptr + 1

        _quicksort(array, low, final_pivot_index - 1)
        _quicksort(array, final_pivot_index + 1, high)


@benchmark
def quicksort(array):
    """
    Quick sort function that sort the array in ascending order
    """
    _quicksort(array, 0, len(array) - 1)
    return array


# ------------Heap sort--------------


@njit
def reorder_min_single_node(array, root_index, current_size):
    """
    Reorder a single node using current_size pointer instead of slicing.
    """
    smallest = root_index
    left = 2 * root_index + 1
    right = 2 * root_index + 2

    if left < current_size and array[left] < array[smallest]:
        smallest = left
    if right < current_size and array[right] < array[smallest]:
        smallest = right

    if smallest != root_index:
        array[root_index], array[smallest] = array[smallest], array[root_index]
        reorder_min_single_node(array, smallest, current_size)


@njit
def reorder_min_heap(array, root_index, current_size):
    """
    Reorder the min heap tracking the active size.
    """
    if root_index * 2 + 1 < current_size:
        reorder_min_heap(array, root_index * 2 + 1, current_size)
    if root_index * 2 + 2 < current_size:
        reorder_min_heap(array, root_index * 2 + 2, current_size)

    reorder_min_single_node(array, root_index, current_size)


@benchmark
def heap_sort(array):
    """
    Heap sort function that sort the array in ascending order.
    Because we use a MIN heap in-place, putting the smallest elements at the end
    results in a DESCENDING array. We reverse it at the end to make it ASCENDING.
    """
    current_size = len(array)
    reorder_min_heap(array, 0, current_size)

    while current_size > 0:
        # swap the min element with the very end element of the active heap
        array[0], array[current_size - 1] = array[current_size - 1], array[0]

        # decrease the active size of our heap
        current_size -= 1

        # shift it down using pointers
        if current_size > 0:
            reorder_min_single_node(array, 0, current_size)

    # Reversing a NumPy array is practically instant (O(1) view creation)
    return array[::-1]


def warmup_compiler():
    """
    Numba has a 'cold start' penalty where it compiles the C code on the very first run.
    We pass a tiny dummy array through the algorithms first so the compilation time
    isn't counted in your actual benchmark!
    """
    dummy = np.array([3, 1, 2])
    merge_sort(dummy.copy())
    quicksort(dummy.copy())
    heap_sort(dummy.copy())
    print("--- Compiler Warmup Complete ---\n")


def experiment():
    filename = "dataset.txt"
    line_count = 0

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()

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

                print("\n")

                is_equal = (
                    np.array_equal(result1, result2)
                    and np.array_equal(result2, result3)
                    and np.array_equal(result3, result4)
                )

                if not is_equal:
                    print("NOT EQUAL !")
                    continue

                print(f"result:  {is_equal}\n")

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
    warmup_compiler()
    experiment()


if __name__ == "__main__":
    main()
