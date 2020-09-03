import torch
from tqdm import tqdm
import os

def save():

	exampleList = []
	cached_features_file = os.listdir("./owt_tensor")
	
	for i in tqdm(range(len(cached_features_file))):
		_file = cached_features_file[i]
		exampleList += torch.load(os.path.join("./owt_tensor/", _file))

	print("saving..")
	torch.save(exampleList, "cached_lm_BertTokenizer_128_owt_txt")

def save_1000000():
	_cached_features_files = os.listdir("./owt_tensor/")
	addfile = []

	for _file in _cached_features_files:
		startidx = _file.split('_')[-2]
		endidx = _file.split('_')[-1]
		print(f"startidx:{startidx}, endidx: {endidx}")
		idx = 1
		if int(startidx) >= 1000000*idx and int(endidx) <= 1000000*(idx+1):
			addfile += torch.load(os.path.join("owt_tensor", _file))
	print("saving...")
	torch.save(addfile, os.path.join("./owt_tensor_nsml/", "cached_lm_BertTokenizer_128_owt_txt_0_1000000"))

def load():
	cached_features_file = "./cached_lm_BertTokenizer_128_owt_txt"
	print(torch.load(cached_features_file))


if __name__ == "__main__":
	save_1000000()
