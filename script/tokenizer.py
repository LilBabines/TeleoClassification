from transformers import PreTrainedTokenizer
from itertools import product
import json

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




if __name__ == '__main__':
    k = 3
    tokenizer = KmerTokenizer(k=k)
    sequence = "ATCGGCTA"
    tokens = tokenizer.tokenize(sequence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    save_directory = './'
    print(tokenizer.get_vocab())
