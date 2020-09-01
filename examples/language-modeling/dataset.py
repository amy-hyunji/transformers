import logging
import os
import tarfile
import pickle
import time
from tqdm import tqdm
import sys

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

#from transformers.tokenization_utils import PreTrainedTokenizer
import multiprocessing
from pathos.multiprocessing import ProcessingPool

import math
import platform
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
	CONFIG_MAPPING,
	MODEL_WITH_LM_HEAD_MAPPING,
	AutoConfig,
	AutoModelWithLMHead,
	AutoTokenizer,
	DataCollatorForLanguageModeling,
	HfArgumentParser,
	PreTrainedTokenizer,
	Trainer,
	TrainingArguments,
	set_seed,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MDOEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

"""
Loader function to solve MemoryError
File format: .xz files inside a file
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

		startidx = 0
		endidx = 10000

		# ./ -> directory
		cached_features_file = os.path.join(
			"./", "cached_lm_{}_{}_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename, startidx, endidx),
		)

		xzFiles = os.listdir(self.file_path)
		self.txtpath = "./owt_txt"
		self.tensorpath = "./owt_tensor"	
		"""
		if not os.path.exists(self.txtpath):
			os.mkdir(self.txtpath)
		print("Dumping txt file to owt_txt")
		for i in tqdm(range(len(xzFiles))):
			xzfile = xzFiles[i]
			with tarfile.open(os.path.join(self.file_path, xzfile)) as f:
				f.extractall(self.txtpath)
		"""

		fileList = os.listdir(self.txtpath)
		fileList.sort()
		fileList = fileList[startidx:endidx]
		logger.info(f"start index: {startidx}, end index: {endidx}")
		logger.info(f"{len(fileList)} number of file exists: should be {endidx-startidx+1}")
		self.examples = list()
		
		if os.path.exists(cached_features_file) and not overwrite_cache:
			logger.info(f"Loading features from {cached_features_file}")
			self.examples = torch.load(cached_features_file)
		else:
			logger.info("Tokenizing the Dataset")
			self._tokenize(fileList)
			logger.info("Saving....")	
			torch.save(self.examples, cached_features_file)
			logger.info("Done Saving....")		
			sys.exit()

	def _tokenize(self, file_list):
		for i in tqdm(range(len(file_list))):
			_file = file_list[i]
			fileList = list()
			tokenized_text = list()
			f = open(os.path.join(self.txtpath, _file), "r")
			lines = f.readlines()
			for line in lines:
				text = line.rstrip('\n')
				if (text == ""): continue
				tokenized_text += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
				if len(tokenized_text) > self.block_size:
					self.examples.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokenized_text[:self.block_size]), dtype=torch.long))
					fileList.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokenized_text[:self.block_size]), dtype=torch.long))
					tokenized_text = tokenized_text[self.block_size:]
			f.close()
		return 
	
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return torch.tensor(self.examples[i], dtype=torch.long)
	
"""
Loader function to solve MemoryError
File format: .xz files inside a file
MultiProcess
"""
class MP_SplitTextDataset(Dataset):

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
		self.tensorpath = "./owt_tensor"
		os.makedirs(self.tensorpath, exist_ok=True)
		self.cached_features_file = os.path.join(
			self.tensorpath, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename),
		)
		
		num_cores = 10 #int(multiprocessing.cpu_count() / 2)
	
		self.txtpath = "./owt_txt"
		if not os.path.exists(self.txtpath):
			os.mkdir(self.txtpath)
			print("Dumping txt file to owt_txt")
			xzFiles = os.listdir(self.file_path)
			for i in tqdm(range(len(xzFiles))):
				xzfile = xzFiles[i]
				with tarfile.open(os.path.join(self.file_path, xzfile)) as f:
					f.extractall(self.txtpath)
			xzFiles = None
			del(xzFiles)

		fileList = os.listdir(self.txtpath)
		fileList.sort()
		logger.info(f"{len(fileList)} number of file exists")
		file_per_core = int(len(fileList)/num_cores)
		split_file = list()
		for i in range(num_cores):
			if (i == num_cores-1):
				split_file.append(fileList[file_per_core*i:])
			else:
				split_file.append(fileList[file_per_core*i:file_per_core*(i+1)])
		fileList = None
		del(fileList)
		self.examples = list()
		
		with FileLock(self.cached_features_file+".lock"):
			if os.path.exists(self.cached_features_file) and not overwrite_cache:
				logger.info(f"Loading features from {self.cached_features_file}")
				self.examples = torch.load(self.cached_features_file)
			else:
				pool = ProcessingPool(nodes=num_cores)
				temp = pool.map(self._tokenize, split_file)
				logger.info("Done Saving all!!")
				sys.exit()
		
	def _tokenize(self, file_list):
		p_feature_file = self.cached_features_file + f"_{os.getpid()}"
		for i in range(len(file_list)):
			if i%10000 == 0:
				print(f"pid: {os.getpid()} is working on {i}/{len(file_list)}")
				if i > 0:
					print(f"Saving pid {os.getpid()}. Dumping list of length {len(defList)}")
					torch.save(defList, p_feature_file+"_"+str(i))
				defList = None
				del defList
				defList = list()
			_file = file_list[i]
			tokenized_text = list()
			f = open(os.path.join(self.txtpath, _file), "r")
			lines = f.readlines()
			for line in lines:
				text = line.rstrip('\n')
				if (text == ""): continue
				tokenized_text += self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
				if len(tokenized_text) > self.block_size:
					defList.append(torch.tensor(self.tokenizer.build_inputs_with_special_tokens(tokenized_text[:self.block_size]), dtype=torch.long))
					tokenized_text = tokenized_text[self.block_size:]
			lines = None
			del lines
			f.close()
		
		print(f"Saving pid {os.getpid()}. Dumping list of length {len(defList)}")
		torch.save(defList, p_feature_file)
		print(f"Done saving pid {os.getpid()}")
		return 
	
	def __len__(self):
		return len(self.examples)

	def __getitem__(self, i) -> torch.Tensor:
		return torch.tensor(self.examples[i], dtype=torch.long)
	
class BertTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False,
    ):
          start = time.time()
          self.examples = torch.load(file_path)
          logger.info(
               "Loading features from cached file %s [took %.3f s]", file_path, time.time() - start
           )

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

        if os.path.exists(cached_features_file) and not overwrite_cache:
             start = time.time()
             self.examples = torch.load(cached_features_file)
             logger.info(
                  "Loading features from cached file %s [took %.3f s]", cached_features_file, time.time() - start
             )


        else:
#           lock_path = cached_features_file + ".lock"
#           with FileLock(lock_path):
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = list()
            logger.info(f"Opening: {file_path}")
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
