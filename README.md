# Multi-Group Parity (MGP) 

This is official implementation of [Generalised Bias Mitigation for Personality Computing]().

 In this work, we design a novel fairness loss function named Multi-Group Parity (MGP) to provide a generalised
approach for bias mitigation in personality computing. In contrast to existing works in the literature, MGP is generalised as it features four ‘multiple’ properties (4Mul): multiple tasks, multiple modali-ties, multiple sensitive attributes, and multi-valued attributes. Exten-sive experiments on two large multi-modal benchmark personality computing datasets demonstrate that the MGP sets new state-of-the-art performance both in the traditional and in the proposed 4Mul settings.

## Prerequisites
This project is build upon Pytorch Lightning. 


Create a new python env named **mgp**
from requirements.txt
```
$ pip install -r requirements.txt

```
## Prepare Benchmark Datasets

UDIVA benchmark (multiple modality features are Extracted and Aligned ) is available at [training_set](https://drive.google.com/file/d/197Br6CDmoNmx41cVa-NfVd5OviNDa95n/view?usp=sharing) and [test_set](https://drive.google.com/file/d/1BV-1CahnplE0kFr0xN3FIulkypLj9gvt/view?usp=drive_link)

FIV2 benchmark (multiple modality features are Extracted and Aligned ) is available at [training_set](https://drive.google.com/file/d/192dBNGKLUFyaxNEQAwrDtbuuvxiv0FFA/view?usp=drive_link) and [test_set](https://drive.google.com/file/d/1O_2m4IhnBw6ZxbsA-ZKZe1-nV8xnD_3L/view?usp=drive_link)



and put them in the root directory of the project.

## Training

configs folder contains the configuration files for training and testing.


