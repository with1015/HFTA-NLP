import torch
import spacy
import argparse
import random
import time
import math

from model import Seq2Seq
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

from hfta.optim import get_hfta_optim_for

parser = argparse.ArgumentParser(description="Seq2Seq Training Argument parser.")

parser.add_argument('--batch-size', default=128, type=int,
                    help='Batch size to train')
parser.add_argument('--epochs', default=100, type=int,
                    help='Training epochs')
parser.add_argument('--lr', default=0.1, type=float,
                    help='Training learning rate')
parser.add_argument('--source-dataset', default=None, type=str,
                    help='Source dataset (ex. de_core_news_sm)')
parser.add_argument('--target-dataset', default=None, type=str,
                    help='Target dataset (ex. en_core_web_sm)')
parser.add_argument('--fusion-size', default=1, type=int,
                    help='Determine fused array size B')
parser.add_argument('--print-freq', default=10, type=int,
                    help='Print step frequency during training')

args = parser.parse_args()

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spacy_src = spacy.load(args.source_dataset)
spacy_tgt = spacy.load(args.target_dataset)

def tokenize_src(text):
    return [token.text for token in spacy_src.tokenizer(text)[:-1]]

def tokenize_tgt(text):
    return [token.text for token in spacy_tgt.tokenizer(text)]

src = Field(tokenize=tokenize_src,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

tgt = Field(tokenize=tokenize_tgt,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(src, tgt))

print("[INFO] Training dataset expected:", len(train_data.examples))
print("[INFO] Validation dataset expected:", len(valid_data.examples))
print("[INFO] Test dataset expected:", len(test_data.examples))

src.build_vocab(train_data, min_freq=2)
tgt.build_vocab(train_data, min_freq=2)

print("[INFO] Unique tokens in source vocabulary:", len(src.vocab))
print("[INFO] Unique tokens in target vocabulary:", len(tgt.vocab))

batch_size = args.batch_size

train_loader, valid_loader, test_loader = BucketIterator.splits((train_data, valid_data, test_data),
                                                                batch_size=batch_size,
                                                                device=device)

print("[INFO] Creating Seq2Seq model...")

model = Seq2Seq(input_size=len(src.vocab),
                output_size=len(tgt.vocab),
                hidden_size=512,
                n_layers=2,
                enc_emb_dim=256,
                dec_emb_dim=256,
                B=args.fusion_size).to(device)

tgt_pad_idx = tgt.vocab.stoi[tgt.pad_token]
criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = get_hfta_optim_for(torch.optim.Adam, B=args.fusion_size)(model.parameters(), lr=args.lr)

model.train()

for epoch in range(args.epochs):
    print(f'[INFO] epoch [{epoch}/{args.epochs}]')
    for idx, batch in enumerate(train_loader):
        step_start = time.time()
        #src = batch.src.to(device)
        #tgt = batch.trg.to(device)

        # Synthetic data -> avoid data loading bottleneck
        src = torch.randint(10, (26, args.batch_size), device=device)
        tgt = torch.randint(10, (26, args.batch_size), device=device)

        optimizer.zero_grad()

        fw_start = time.time()
        output = model(src, tgt)
        fw_end = time.time()

        result = output[0][1:].view(-1, output[0].shape[-1])
        tgt = tgt[1:].view(-1)

        bw_start = time.time()
        loss = criterion(result, tgt)
        loss.backward()
        bw_end = time.time()

        opt_start = time.time()
        optimizer.step()
        step_end = time.time()

        if idx % args.print_freq == 0:
            fw_time = fw_end - fw_start
            bw_time = bw_end - bw_start
            opt_time = step_end - opt_start
            step_time = step_end - step_start
            print(f"[INFO] step {idx} time: {step_time} | FW {fw_time} | BW {bw_time} | OPT {opt_time}")
