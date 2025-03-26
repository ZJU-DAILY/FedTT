# FedTT

In this paper, we propose FedTT, an effective, efficient, privacy-aware cross-city traffic knowledge transfer framework,
which transforms the traffic domain of data from source cities and trains the traffic model on transformed data in the
target city. First, we design a traffic view imputation method to complete and predict missing traffic data by capturing
spatio-temporal dependencies of traffic view. Second, we utilize a traffic domain adapter to uniformly transform the
traffic data from the traffic domains of source cities into that of the target city. Third, we propose a traffic secret
transmission method to securely transmit and aggregate the transformed data using a lightweight traffic secret
aggregation method. Fourth, we introduce a federated parallel training method to enable the simultaneous training of
modules. Experiments using 4 datasets on three mainstream traffic prediction tasks demonstrate the superiority of the
framework. In future work, we aim to extend FedTT to support a broader range of spatio-temporal prediction tasks.

![framework.png](framework.png)

## Environment

Python.version = 3.9.21<br>
Other dependencies are listed in requirements.txt.

All the experiments are conducted in the federated environment on four nodes, one as a server and the other three as
clients, each equipped with two Intel(R) Xeon(R) CPU E5-2650 v4@2.20GHz 12-core processors, 128GB of RAM, and a
Internet speed of 100MB/s.

## Datasets

The datasets used in this paper are public and available
in https://drive.google.com/drive/folders/1a4JNMJNWYGF4he7rlt_aYT3I4P4XX8FK?usp=sharing.

## Complication

The running example of FedTT is as follows.

````
python main.py --address 'localhost' --batch_size 128 --clients_num 3--device 'cuda' --epoch 100 --frozen 5 --learning_rate 0.005 --model 'mlp' --task 'flow' --test_split 0.1 --time_step 12 --train_split 0.8 --val_split 0.1

