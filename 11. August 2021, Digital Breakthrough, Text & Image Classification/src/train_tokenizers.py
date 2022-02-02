from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from pathlib import Path


class TokenizerFabrica(object):

    def __init__(self, 
                 type,
                 pre_tokenizer,
                 normalizer,
                 tokenizer_args,
                 trainer_args):
        if type == 'WordPieceTrainer':
            self.trainer = WordPieceTrainer(**trainer_args)
            self.tokenizer = Tokenizer(WordPiece(**tokenizer_args))

        elif type == 'BpeTrainer':
            self.trainer = BpeTrainer(**trainer_args)
            self.tokenizer = Tokenizer(BPE(**tokenizer_args))

        elif type == 'UnigramTrainer':
            self.trainer = UnigramTrainer(**trainer_args)
            self.tokenizer = Tokenizer(Unigram(**tokenizer_args))

        else:
            raise ValueError(f'unknown tokenizer: {type}')

        pre_tokenizers = []
        for pre_tkn in pre_tokenizer:
            if pre_tkn == 'whitespace':
                pre_tokenizers.append(Whitespace())
            elif pre_tkn == 'digits':
                pre_tokenizers.append(Digits())
            else:
                raise ValueError(f'unknown pre_tokenizer: {pre_tkn}')

        self.tokenizer.pre_tokenizer = Sequence(pre_tokenizers)
        if normalizer == 'lowercase':
            self.tokenizer.normalizer = Lowercase()
        else:
            raise ValueError(f'unknown normalizer: {normalizer}')

        self.fitted = False
    
    def fit(self, item_names):
        self.tokenizer.train_from_iterator(item_names, 
                                           self.trainer)
        self.fitted = True

    def save_model(self, output_path: Path):
        if self.fitted:
            self.tokenizer.save(str(output_path))
        else:
            raise ValueError('Fit tokenizer before saving')
