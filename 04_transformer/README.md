
## quick start
 - (1) download the [tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset, tokenize it with the GPT-2 Tokenizer, 

```bash
pip install -r requirements.txt
python prepro_tinyshakespeare.py
python train_gpt2.py
```

输出为：
```text
Saved 32768 tokens to data/tiny_shakespeare_val.bin
Saved 305260 tokens to data/tiny_shakespeare_train.bin
```


 - (2) download and save the GPT-2 (124M) weightsit will save three files: 
    - 1) the gpt2_124M.bin file that contains the raw model weights for loading in C, 
    - 2) the gpt2_124M_debug_state.bin, which also contains more debug state: the inputs, targets, logits and loss (useful for debugging and unit testing), and finally 
    - 3) the gpt2_tokenizer.bin which stores the vocabulary for the GPT-2 tokenizer, translating token ids to byte sequences of UTF-8 encoded string pieces. 

 ```bash
python train_gpt2.py
```


- (3) init from them in C/CUDA and train for one epoch on tineshakespeare with AdamW
    - compile with CUDA (without cuDNN), and also turn on flash attention V1. 
```bash
make train_gpt2cu
./train_gpt2cu
```
- 
    - compile with cuDNN, and also go faster with flash attention V2. 

```
make train_gpt2cu USE_CUDNN=1
./train_gpt2cu
```
- 
    - set batch size (default is 4)
```
./train_gpt2cu -b 32
```
- 
    - Multi-GPU training. Make sure you install MPI, e.g. on Linux:

```
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev
```

and then:
```
make train_gpt2cu
mpirun -np <number of GPUs> ./train_gpt2cu
```
