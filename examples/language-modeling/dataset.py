import logging
import os
import pickle
import time
from tqdm import tqdm
import sys

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer
import multiprocessing
from pathos.multiprocessing import ProcessingPool

logger = logging.getLogger(__name__)

"""
Loader function to solve MemoryError
"""
class SplitTextDataset(Dataset):

	def __init__(
		self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache: False,
	):
		self.tokenizer = tokenizer
		self.file_path = file_path
		self.overwrite_cache = overwrite_cache
		self.full_block_size = block_size
		self.block_size = block_size
		self.block_size -= tokenizer.num_special_tokens_to_add(pair=False)
		directory, filename = os.path.split(file_path)
		# ./ -> directory
		cached_features_file = os.path.join(
			"./", "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename),
		)
		
		num_cores = multiprocessing.cpu_count()
		print(f"***** number of CORES: {num_cores} *****")
		fileList = os.listdir(self.file_path)
		logger.info(f"{len(fileList)} number of file exists")
		file_per_core = int(len(fileList)/num_cores)
		split_file = list()
		for i in range(num_cores):
			if (i == num_cores-1):
				split_file.append(fileList[file_per_core*i:])
			else:
				split_file.append(fileList[file_per_core*i:file_per_core*(i+1)])
		self.examples = list()
		
		with FileLock(cached_features_file+".lock"):
			if os.path.exists(cached_features_file) and not overwrite_cache:
				logger.info(f"Loading features from {cached_features_file}")
				self.examples = torch.load(cached_features_file)
			else:
				pool = ProcessingPool(nodes=num_cores)
				temp = pool.map(self._tokenize, split_file)
				for _list in temp:
					self.examples += _list
				torch.save(self.examples, cached_features_file)
				sys.exit()
		
	def _tokenize(self, file_list):
		defList = list()
		for i in range(len(file_list)):
			if i%30 == 0:
				print(f"pid: {os.getpid()} is working on {i}/{len(file_list)}")
			_file = file_list[i]
			tokenized_text = list()
			f = open(os.path.join(self.file_path, _file), "r")
			lines = f.readlines()
			for line in lines:
				text = line.rstrip('\n')
				if (text == ""): continue
				tokenized_text += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
				if len(tokenized_text) > self.block_size:
					defList.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokenized_text[:self.block_size]), dtype=torch.long))
					tokenized_text = tokenized_text[self.block_size:]
		return defList 
	
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return torch.tensor(self.examples[i], dtype=torch.long)
					
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
        assert os.path.isfile(file_path)
        self.full_block_size = block_size
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                self.examples = torch.load(cached_features_file)
                logger.info(
                    "Loading features from cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

            else:
                logger.info("Creating features from dataset file at %s", directory)
                self.examples = list()
                with open(file_path, encoding="utf-8") as f:
                    tokenized_text = list()
                    lines = f.readlines()
                    for i in tqdm(range(len(lines))):
                        line = lines[i]
                        text = line.rstrip('\n')
                        tokenized_text += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                        if len(tokenized_text)>block_size:
                            self.examples.append(torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenized_text[:block_size]), dtype=torch.long))
                            tokenized_text = tokenized_text[block_size:]

                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                torch.save(self.examples, cached_features_file)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.rstrip('\n') for line in f if (len(line) > 0 and not line.isspace())]

        # print(len(lines))
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


if __name__ == "__main__":
	pass
