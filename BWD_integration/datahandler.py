from torch.utils.data import Dataset, DataLoader
import torch
import io
import numpy as np

class MathData(Dataset):
    """
    This class creates an interface for the dataset for the DataLoader function
    """
    def __init__(self, config, env, dtype):
        super(MathData, self).__init__()
        self.data = []
        self.dtype = dtype
        self.env = env
        self.pad_index = config.pad_index
        self.eos_index = config.eos_index
        if dtype == 'train':
            self.path = config.train_dir
        if dtype == 'valid':
            self.path = config.valid_dir
        if dtype == 'test':
            self.path = config.test_dir
        self.load_data(config)

    def load_data(self, config):
        """
        Loads the data from the disk and put to self.data
        """
        lines = []
        if self.dtype == 'train':
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == config.train_reload_size:
                        break
                    else:
                        lines.append(line.rstrip().split('|'))
        else:
            with io.open(self.path, mode='r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == config.test_reload_size:
                        break
                    else:
                        lines.append(line.rstrip().split('|'))
        self.data = [xy.split('\t') for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        if self.dtype == 'train':
            self.size = 1 << 60
        else:
            self.size = config.test_reload_size

    def __len__(self):
        """
        Retuens the size of the dataset
        """
        return self.size

    def __getitem__(self, idx):
        """
        Returns an item in index = idx from the dataset
        """
        if self.dtype == 'train':
            idx = np.random.randint(len(self.data))
        x, y = self.data[idx]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def create_batch(self, sequences):
        """
        Creates a batch from the given list of equations.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2
        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index
        return sent, lengths

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_ops = [sum(int(word in self.env.OPERATORS) for word in seq) for seq in x]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]
        x, x_len = self.create_batch(x)
        y, y_len = self.create_batch(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)



def create_data_loader(config, env, dtype):
    """
    Returns an iterable to iterate over the data. dtype can be one of the followings:
    'train', 'valid', 'test'
    """
    assert dtype in ['train', 'valid', 'test']
    dataset = MathData(config, env, dtype)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )
