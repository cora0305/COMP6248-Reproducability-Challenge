# COMP6248-Reproducability-Challenge
## SymbolicMathematics

PyTorch original implementation of Deep Learning for Symbolic Mathematics (ICLR 2020).https://arxiv.org/abs/1912.01412

In this task, we show that the machine translation model is suitable for symbolic mathematic tasks, and the seq2seq model can be used for difficult tasks like function integration. We use Forward Generation (FWD) to generate polynomials and express them as sum of powers. We use Backward Generation (BWD) to generate data and calculate the integration. After experiments, we show that the 6th experiment (100,000 epoch size, 32 batch size, 1,000,000 training size) has the best performance with the accuracy of 95.4\%.

### Dataset
BWD_integration: https://dl.fbaipublicfiles.com/SymbolicMathematics/data/prim_bwd.tar.gz

FWD_polynomial: generated with code

### Environment:
BWD: environemnt.yml
FWD: python3.7 pytorch sympy

### How to run with code
1. BWD: Firstly, setting the environment and downloading the dataset with the link. Second, put the data into the correct directory Dataset/Prime_BWD. Then runing the file 'main.py' with the different setting such as 'python main.py --epoch_size 100000 --batch_size 32 --max_epoch 30 --train_reload_size 1000000 --test_reload_size 500'. Finally, check the file in the 'Dumped' to observe the results that in the log file.

2. FWD: Firstly, setting the environment. Then, using the 'data_gen.py' to generated the polynomial dataset. After that, using the 'train.py' to train the model with the different epoch, batch size and so on. Finally, check the logger and result with the 'train.py'.

### Member of group
Xiaoxuan Wu, Ruixue Guo, Wanyuan Lin, Zeliang Zhao
