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


class KmerTokenizer(PreTrainedTokenizer):
    def __init__(self, k,vocab=None, **kwargs):
        
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[MASK]': 3,
            '[UNK]': 4
        }

        

        
        self.k = k

        if vocab is None:

            # self.vocab =
            self.vocab = dict(self.special_tokens, **self.build_kmer_vocab(k))
        else:
            self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(**kwargs)

        self.add_special_tokens({'pad_token': '[PAD]'})
        self.add_special_tokens({'cls_token': '[CLS]'})
        self.add_special_tokens({'sep_token': '[SEP]'})
        self.add_special_tokens({'mask_token': '[MASK]'})
        self.add_special_tokens({'unk_token': '[UNK]'})

        self.vocab_size = self.vocab_size()
    
    def vocab_size(self):
        return len(self.vocab)
    def build_kmer_vocab(self, k):
        nucleotides = ['A', 'C', 'G', 'T']

        kmers = [''.join(p) for p in product(nucleotides, repeat=k)]
        return {kmer: idx+len(self.special_tokens.keys()) for idx, kmer in enumerate(kmers)}

    def _tokenize(self, text):
        return self.generate_kmers(text, self.k)

    def generate_kmers(self, sequence, k):
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get('[UNK]'))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, '[UNK]')

    def get_vocab(self):
        return self.vocab

    def save_vocabulary(self, save_directory):
        vocab_file = f"{save_directory}/vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
        return (vocab_file,)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        cls = [self.vocab['[CLS]']]
        sep = [self.vocab['[SEP]']]

        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return cls + token_ids_0 + sep
            else:
                output = []
                num_pieces = int(len(token_ids_0)//510) + 1
                for i in range(num_pieces):
                    output.extend(cls + token_ids_0[510*i:min(len(token_ids_0), 510*(i+1))] + sep)
                return output

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a token type ID tensor from a pair of sequences for sequence classification tasks.
        """
        sep = [self.vocab['[SEP]']]
        cls = [self.vocab['[CLS]']]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + sep) * [0]
    

class KmerBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file, k, **kwargs):
        super().__init__(vocab_file=vocab_file,cleaup_tokenization_spaces =True, **kwargs)
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
        if not os.path.exists(f"resources/tokenizers/vocab_{name}mer.txt"):
            generate_kmer_vocab(name)
        print("-----------------")
        print(f"Generated {name}-mer vocabulary")
        print("-----------------")
        tokenizer = KmerBertTokenizer(vocab_file=f"resources/tokenizers/vocab_{name}mer.txt", k=name)

       
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

