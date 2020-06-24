This is the implementation of paper **DAve-QN: A Distributed Averaged Quasi-Newton Method with Local
Superlinear Convergence Rate**, accepted at the 23rd International Conference on
Artificial Intelligence and Statistics ([AISTATS-2020](https://www.aistats.org/ )). 

This is a high performance implementation in C using [MPI](https://mpitutorial.com/tutorials/). In order to compare to the state-of-the-art, we have implemented [GIANT](https://papers.nips.cc/paper/7501-giant-globally-improved-approximate-newton-method-for-distributed-optimization.pdf), [DAve-RPG](http://proceedings.mlr.press/v80/mishchenko18a/mishchenko18a.pdf) and [DANE](https://arxiv.org/pdf/1312.7853.pdf) with all needed scripts to run.  

We also provide a MATLAB implementation of DAve-QN for further use.

## Requirements

Intel MKL 11.1.2
MVAPICH2/2.1
mpicc 14.0.2


## Compilation

First we have to set the environment variable MKLROOT. This depends on the path that MKL is installed. For default installation on LINUX systems: 

```sh
$ export MKLROOT=/opt/intel/mkl/
```

then we can compile the code using the provided makefile:
```sh
$ make
```


## Tests

DAve-QN accepts multiple parameters as input. A typical test looks like this:

```sh
$ mpirun -np 3 dave_qn.o /path/to/mnist 60000 9994156 780 40 1 0.01 1
```

This will run dave_qn on `mnist` dataset with 3 processors (2 workers and 1 master, indicated by -np). The input parameters to dave_qn are described as below:

mpirun -np [number of processors] [path] [nrows] [nnz] [ncols] [iterations] [lambda] [gamma] [freq]

path: full path to the dataset
nrows: number of samples in the dataset
nnz: number of non-zeros in the dataset
ncols: number of columns in the dataset
iterations: number if iterations to run
lambda: regularization parameter
gamma: initial step size for better initialization – this should be very small, usually 1e-3 or 1e-4.  
Freq: frequency of computing objective function and printing the output. If you are running the program for many iterations, it is better to set this to higher values like 10, 100 or more depending on iterations.


A simple bash file is provided that can run mnist on 2 workers and one master. In order to run, mnist dataset has to be split in two and put in a directory called dataset next to the code. Therefore,  dataset folder should contain mnist, mnist-0 and mnist-1 which are respectively the main dataset, the first split and the second split. Then, you can simply run the code using:

```sh
$ sh test.sh
```


> **_NOTE:_** you can find all the scripts for tests in the `scripts` directory.

## MATLAB Implementation
We also provide a MATLAB implementation for DAve-QN which you can find it in "MATLAB Code" directory. You will need LIBSVM [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/ ""). 

## How to split a dataset?

We provide a simple jar file that can split a dataset into arbitrary pieces. For example, to split the mnist dataset into 2 parts, you should put mnist in the dataset folder. Then, simply run the following:

```sh
$ java -jar Split.jar /path/to/dataset mnist 60000 2
```

To be more precise, Split.jar accepts the following parameters:

```sh
$ java -jar Split.jar path filename nrows nparts 
```

where path indicates the directory that contains the main dataset, filename is the dataset name, nrows is the number of rows in the dataset and nparts is the number of parts.

> **_NOTE:_** you can find `mnistSplit.sh` in the scripts folder and run `sh mnistSplit.sh`.

## Output

The output of the code contains three columns. First column is the time in milliseconds, second column is the objective function value and third column is the norm of the gradient. 


## Troubleshooting

If you get an error as  `mpirun was unable to find the specified executable file...`, most likely it means that you have not compile the code properly. Make sure you have compiled the code using “make” command without any error.


If you get an error as `File not found!`, this means that one of the needed files for the dataset is not present. Make sure you put the dataset in the proper destination and you split it before running the code. Please refer to **How to split a dataset** section. 
