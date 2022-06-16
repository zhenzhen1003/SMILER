# SMILER
## Structuring Meaningful Code Changes in Developer Community
### Datasets
Paper reference: R. Tufan, L. Pascarella, M. Tufanoy, D. Poshyvanykz, and G. Bavota, “Towards automating code review activities,” in 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE). IEEE, 2021, pp. 163–174.
### Requirements:
- Tensorflow-gpu 1.15.4           **This project should work with GPU. If you don't have a GPU environment, training and inference will be very slow.**
- gensim         4.1.2
#
## Quick setup for training
### 1.Prepare Dataset
The dataset is in the Dataset directory, which contains the source dataset and the processed dataset.Our training code needs to convert the real code into a token form and combine the input and output, separating each code with a -1 flag.
The processing process is shown in the figure：
![Dataset](https://user-images.githubusercontent.com/72842107/174091913-83f5ff64-3acc-477e-bb0e-f453d4c433c6.png)

If you want to use a custom dataset, please refer to the following procedure：
#### a.Merage all files
Now you have `src-xxx.txt` and `tgt-xxx.txt`.You need to merge these file contents into `all.txt`.
For example:
* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`
* `src-test.txt`
* `tgt-test.txt`

Linux:
```bash
cat src-train.txt tgt-train.txt src-val.txt tgt-val.txt src-test.txt tgt-test.txt > all.txt
```
Windows:
```bash
type src-train.txt tgt-train.txt src-val.txt tgt-val.txt src-test.txt tgt-test.txt > all.txt
```
#### b.Create Vocab
For the generated `all.txt`, execute `create_vocab.py` to generate the vocabulary.
#### c.Generate
For `src-xxx.txt` and `tgt-xxx.txt`, you can execute `create_dataset.py` to generate a file in Token format.
