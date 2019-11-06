import re
import numpy as np
import pandas as pd
import os
import warnings
import itertools
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_cddd.utils import transform_smiles, get_descriptors

TOKENS = [
    'O', 'c', '1', '(', 'C', '2', 'o', ')', 'N', 'n', 's', 'p',
    'n', '-', '3', '=', '4', 'F', 'S', 'Cl', 'Br', '[', 'b',
    'H', ']', '5', '+', '#', 'I', '6', 'B', 'P', 'Si', 'si',
    'H3', 'se', 'Se', 'H2', '_', '7', '8', '9',
    'SOS', "EOS", "UNK", "PAD",
]
PADVALUE = TOKENS.index("PAD")
REGEX_SEQ1 = r'(\[[^\[\]]{2,6}\]|Cl|Br|%[\d]{2}|\d|[#\)\(\+\-=CBFIHONPScionps])'
REGEX_SEQ2 = r'([A-Z][a-z]?|[a-z]{1,2})(H\d?)?([+-]\d?)?'


class SmilesTokenizer(object):
    """
    Class that handles transformation between SMILES sequences and one-hot encodings.
    """
    def __init__(self):
        self.tokens = TOKENS
        self.regex1 = REGEX_SEQ1
        self.regex2 = REGEX_SEQ2

    def tokenize_sequence(self, sequence):
        """
        Method to tokenize SMILES strings.
        :param sequence: SMILES sequence to tokenize.
        :return: List with tokenized SMILES
        """
        new_token_list = []
        token_list = list(filter(None, re.split(self.regex1, sequence)))
        for token in token_list:
            if "[" in token:
                new_token_list.append("[")
                new_token_list.extend(self._tokenize_inside_brackets(token[1:-1]))
                new_token_list.append("]")
            elif token == "-":
                # exchange "-" for single bonds to "_"
                new_token_list.append("_")
            elif '%' in token:
                new_token_list.append("UNK")
            else:
                new_token_list.append(token)
        return new_token_list

    def _tokenize_inside_brackets(self, sequence):
        """
        Method that that tokenizes the SMILES sequence inside squared brackets
        to handle non-standard atoms, charges, etc.
        :param sequence: Content inside squared brackets of a SMILES string.
        :return: List with tokens.
        """
        token_list = list(filter(None, re.split(self.regex2, sequence)))
        token_list = [token if token in self.tokens else "UNK" for token in token_list]
        """if "UNK" in token_list:
            warnings.warn("Warning: Unknown token encountered in the bracket [{}]".format(sequence))"""
        return token_list

    def _one_hot_encode(self, token_list):
        """
        Method that one-hot encodes a list of tokens.
        :param token_list: List with tokens.
        :return: Numpy array with one-hot encoded tokens. [len(token_list), number of unique tokens]
        """
        one_hot = []
        for token in token_list:
            one_hot.append(list(map(lambda s: token == s, self.tokens)))
        return np.array(one_hot).astype(np.int)

    def untokenize_list(self, token_list):
        """
        Method to create a SMILES sequence from a list of tokens.
        :param token_list: List of tokens.
        :return: SMILES sequence.
        """
        sequence = "".join(token_list).replace("_", "-",)
        return sequence

    def encode(self, sequence, one_hot=False, start_token=False, end_token=False):
        """
        Method that tokenizes a SMILES sequence and one-hot encodes it in an array.
        :param sequence: SMILES sequence
        :param one_hot: If True, one-hot encodes the sequence.
        :param start_token: if True, adds start token to the beginning.
        :param end_token: if True, adds end token to the end.
        :return: One-hot encoded array.
        """
        token_list = self.tokenize_sequence(sequence)
        if start_token:
            token_list = ['SOS'] + token_list
        if end_token:
            token_list = token_list + ['EOS']
        if one_hot:
            output = self._one_hot_encode(token_list)
        else:
            output = np.array([TOKENS.index(token) for token in token_list])
        return output

    def decode(self, array, one_hot=True, stop_at_eos=True, props=False):
        """
        Method that takes a encoded SMILES sequence and transforms it back to a SMILES sequence.
        :param array: encoded array.
        :param one_hot: Bool whether input array is one-hot encoded or holds predicted values. In the latter argmax is
        applied.
        :param stop_at_eos: Terminate decoding at end-of-sequence (EOS) token.
        :return: SMILES sequence.
        """
        if one_hot:
            idx_list = np.where(array == 1)[1].tolist()
        else:
            if props:
                idx_list = np.argmax(array, axis=-1).tolist()
            else:
                idx_list = array.tolist()
        token_list = []
        for idx in idx_list:
            token = self.tokens[idx]
            if token == "EOS" and stop_at_eos:
                break
            elif token in ["SOS", "PAD", "EOS"]:
                continue
            else:
                token_list.append(token)
        sequence = self.untokenize_list(token_list)
        return sequence


class SmilesDataset(Dataset):
    """
    PyTorch Dataset class for loading and one-hot encoding SMILES.
    """
    def __init__(self, data_frame, shuffle=False, smiles_col_header="smiles",
                 input_type="random", output_type="canonical", label_col_header=None, rdkit_descriptors=True):
        super().__init__()
        self.smiles_df = data_frame
        if shuffle:
            self.smiles_df = self.smiles_df.sample(frac=0).reset_index(drop=True)
        self.smiles_col_header = smiles_col_header
        self.input_type = input_type
        self.output_type = output_type
        self.tokenizer = SmilesTokenizer()
        self.transform = NormalizeLabels()
        self.label_col_header = label_col_header
        self.rdkit_descriptors = rdkit_descriptors
        if label_col_header is not None:
            assert not rdkit_descriptors

    def __len__(self):
        return len(self.smiles_df)

    def __getitem__(self, idx):
        smiles = self.smiles_df.loc[idx, self.smiles_col_header]
        input_smiles = transform_smiles(smiles, self.input_type)
        output_smiles = transform_smiles(smiles, self.output_type)
        input_tokens = self.tokenizer.encode(input_smiles, start_token=True, end_token=True)
        output_tokens = self.tokenizer.encode(output_smiles, start_token=True, end_token=True)
        input_sample = torch.from_numpy(input_tokens).float()
        output_sample = torch.from_numpy(output_tokens).float()
        if (self.label_col_header is None) & self.rdkit_descriptors:
            label = torch.Tensor(get_descriptors(smiles))
            label = self.transform(label)
        elif self.label_col_header is not None:
            label = torch.Tensor([self.smiles_df.loc[idx, self.label_col_header]])
        else:
            label = None
        return input_sample, output_sample, label


class SmilesDataLoader(DataLoader):
    """
    PyTorch data loader class that loads batches of padded one-hot encoded SMILES.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, pad_value=PADVALUE, **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda data_list: padded_batch_from_list(data_list, pad_value),
            **kwargs)


def padded_batch_from_list(data_list, pad_value):
    """
    Collate function for PyTorch data loader.
    :param data_list: list of (input_samples, out_samples, label) tuples
    :param pad_value: value used for padding the array of variable length sequences
    :return:
    """
    input_samples = [data[0] for data in data_list]
    output_samples = [data[1] for data in data_list]
    labels = torch.stack([data[2] for data in data_list])

    input_length = torch.as_tensor([len(sample) for sample in input_samples]).long()
    input_length, sorted_indices = torch.sort(input_length, descending=True)
    input_tensor = pad_sequence(input_samples, batch_first=True, padding_value=pad_value).long()
    input_tensor = input_tensor.index_select(0, sorted_indices)
    output_tensor = pad_sequence(output_samples, batch_first=True, padding_value=pad_value).long()
    output_tensor = output_tensor.index_select(0, sorted_indices)
    labels = labels.index_select(0, sorted_indices)

    return input_tensor, input_length, output_tensor, labels, sorted_indices


class NormalizeLabels(object):
    def __init__(self):
        self.mean = torch.Tensor([3.2850, 106.4718, 1.7443, 5.5747, 1.1343, 144.7654, 78.3403])
        self.std = torch.Tensor([1.4116, 17.9807, 0.3520, 1.8585, 0.9497, 23.4297, 27.0827])

    def __call__(self, label_tensor):
        label_tensor = (label_tensor - self.mean) / self.std
        return label_tensor


def make_data_loader(csv, batch_size, num_workers=0, drop_last=False, nrows=None, sampler=None, **kwargs):
    df = pd.read_csv(csv, nrows=nrows)
    dataset = SmilesDataset(df, **kwargs)
    if sampler is "infinit":
        sampler = InfinitSampler(df)
    dataloader = SmilesDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        sampler=sampler
    )
    return dataloader


def batch_to_device(batch, device):
    input_tensor = batch[0].to(device)
    input_length = batch[1].to(device)
    target_tensor = batch[2].to(device)
    labels = batch[3].to(device)
    return input_tensor, input_length, target_tensor, labels


class InfinitSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return itertools.cycle(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

