import random


def generate_dataset(filename="dataset.txt"):
    """
    Generate a dataset file with multiple arrays of random integers and floats.
    args:
        filename is the name of the file to write the dataset to
    return:
        None
    """
    # the dataset will contain 10 arrays, each with 1 million elements 
    # the first array is ascending, the second is descending
    # the next 3 arrays are random integers, the last 5 arrays are random floats

    # I use special mecanisim in the first 2 arrays to avoid memory issue when generating large dataset

    num_arrays = 10
    elements_per_array = 1000000  # 1 million per line

    # Range for 32-bit signed integers
    min_val = -(2**31)
    max_val = 2**31 - 1

    print(f"Starting generation of {filename}...")

    with open(filename, "w") as f:
        for i in range(num_arrays):
            # for memory covention, I will directly write the data to the file

            # first array ascending
            if i == 0:
                init_val = min_val
                for j in range(elements_per_array):
                    init_val += random.randint(0, 100)
                    f.write(f"{init_val} ")
                f.write("\n")

            # second array descending
            elif i == 1:
                init_val = max_val
                for j in range(elements_per_array):
                    init_val -= random.randint(0, 100)
                    f.write(f"{init_val} ")
                f.write("\n")
            else:
                # the first 5 lines are random int and the second 5 lines are random float
                for j in range(elements_per_array):
                    if i < 5:
                        f.write(f"{random.randint(min_val, max_val)} ")
                    else:
                        f.write(f"{random.uniform(min_val, max_val)} ")

                f.write("\n")


if __name__ == "__main__":
    generate_dataset()
