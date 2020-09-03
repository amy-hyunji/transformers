from pathos.multiprocessing import ProcessingPool
import os
import sys

def tokenize(file_list):
	print(f"pid: {os.getpid()} has {len(file_list)} number of files")

num_cores = 3
pool = ProcessingPool(nodes=num_cores)
files = [1, 2, 3, 4, 5, 6, 7, 8]
split_file = list()
file_per_core = int(len(files)/num_cores)

for i in range(num_cores):
	if (i == num_cores-1):
		split_file.append(files[file_per_core*i:])
	else:
		split_file.append(files[file_per_core*i:file_per_core*(i+1)])

temp = pool.map(tokenize, split_file)
print("Done")
sys.exit()
