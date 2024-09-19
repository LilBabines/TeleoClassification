from transformers import AutoTokenizer,PreTrainedTokenizerFast,PreTrainedTokenizer
from itertools import product
import json
from transformers import BertTokenizer
import os

def generate_kmer_vocab(k,save_txt=True,add_1_2_lenght=True):
    nucleotides = ['A', 'T', 'C', 'G']

    kmer_tuples = []
    for i in range(1,k+1)[::-1]:

        kmer_tuples = kmer_tuples + list(product(nucleotides, repeat=i))


    kmer_strings = [''.join(kmer) for kmer in kmer_tuples]
    vocab = {kmer: idx for idx, kmer in enumerate(kmer_strings)}
    print(len(vocab))
    # print(vocab)
    if save_txt:
        with open(f"resources/vocab_{k}mer.txt", "w") as f:

            f.write("[PAD]\n")
            f.write("[CLS]\n")
            f.write("[SEP]\n")
            f.write("[MASK]\n")
            f.write("[UNK]\n")

            for kmer, idx in sorted(vocab.items(), key=lambda item: item[1]):
                f.write(kmer + "\n")

    return vocab

class KmerBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, k, **kwargs):
        super().__init__(vocab_file=vocab_file, **kwargs)
        self.k = k

    def tokenize(self, text, **kwargs):
        # Implement K-mer tokenization
        kmers = [text[self.k*i:self.k*(i+1)] for i in range(len(text)//self.k)]
        if len(text) % self.k != 0:
            kmers.append(text[-(len(text) % self.k):])
        
        # Handle the case where a token might not be in the vocabulary
        tokens = [kmer if kmer in self.vocab else '[UNK]' for kmer in kmers]
        
        return tokens

   
def load_tokenizer(name="zhihan1996/DNABERT-2-117M"):
    '''Load the tokenizer from the model name or the kmer size
    Args:
        name (str or int): The model name or the kmer size
    Returns:
        tokenizer (PreTrainedTokenizerFast): The tokenizer
    '''
    if isinstance(name, str):

        if name=="bpe":

            tokenizer = PreTrainedTokenizerFast(tokenizer_file="resources/tokenizers/teleo_4096.json")

            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            tokenizer.add_special_tokens({'cls_token' : '[CLS]'})
            tokenizer.add_special_tokens({'unk_token' : '[UNK]'})

        else :
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

     
    elif isinstance(name, int):
        if not os.path.exists(f"resources/vocab_{name}mer.txt"):
            generate_kmer_vocab(name)
        tokenizer = KmerBertTokenizer(vocab_file=f"resources/vocab_{name}mer.txt", k=name)

       
    return tokenizer 


from transformers import BertTokenizer

# Example usage:


if __name__ =="__main__":
    # vocab = generate_kmer_vocab(3)

    custom_tokenizer = KmerBertTokenizer(vocab_file="resources/vocab_3mer.txt", k=3,
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                sep_token="[SEP]",
                                                mask_token="[MASK]",
                                                unk_token="[UNK]",
                                                do_lower_case=False,
                                                )
                             
    
    seq= "ATCG"
    print( custom_tokenizer.tokenize(seq))
    input = custom_tokenizer(seq, return_tensors="pt")
    print(input)
    dec = custom_tokenizer.decode(input['input_ids'][0])
    print(dec)
    print( '----------------')

    def tokenize_sequence(sequence, k):
        return [sequence[i:i+k] for i in range(0, len(sequence) - k + 1)]

# Tokenize with K-mer splitting
    k = 3
    kmer_sequence = tokenize_sequence("ATCGT", k)
    encoded_input = custom_tokenizer.encode(" ".join(kmer_sequence), add_special_tokens=True)

    print("Encoded input IDs:", encoded_input)
    # print(custom_tokenizer.tokenize(["ATCGATCGATCGATCG"]))


    print('----------------')
    
    k = 3  # Example K-mer length
    tokenizer = KmerBertTokenizer(vocab_file="resources/vocab_3mer.txt", k=k)

    # Tokenize a sequence
    sequence = "GTAATCCGTACGTACGTTTTGGGGGACGT"
    tokens = tokenizer.tokenize(sequence)
    input = tokenizer(sequence, return_tensors="pt")
    print("Tokens:", input)
    print("Decoded tokens:", tokenizer.decode(input['input_ids'][0]))

