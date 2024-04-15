#!/usr/bin/env python3

import argparse, sys
from os.path import isdir, isfile, join, basename, dirname
from os import listdir

import torch
from scipy.io.wavfile import write

from utils import load_checkpoint, get_hparams_from_file
from models import SynthesizerTrn


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default=None, help='model configuraton')
parser.add_argument('--checkpoint', '-m', type=str, default=None, help='model checkpoint')
parser.add_argument('--vocab', '-v', type=str, default=None, help='vocabulary/symbols list (symbol per line)')
parser.add_argument('--bundle', '-b', '-M', type=str, default=None, help='path to model bundle (as with Meta TTS)')
parser.add_argument('--text', '-t', type=str, default='VITS is truly Awesome!', help='input text')
parser.add_argument('--output', '-o', type=str, default='output.wav', help='output wav')
parser.add_argument('--cpu', '-C', action='store_true', help='CPU only')
parser.add_argument('--speaker', '-s', type=int, default=None, help='select speaker')

args = parser.parse_args()

if not args.bundle and not args.checkpoint and not args.config:
    print(f'error: at least a bundle or checkpoint with configuratoin must be specified')
    sys.exit(1)


if args.checkpoint and not args.config and not args.vocab:
    if isdir(args.checkpoint):
        # use as bundle path
        args.bundle = args.checkpoint
        args.checkpoint = None
    elif isfile(args.checkpoint):
        # check for bundle
        bundle = dirname(args.checkpoint)
        if isfile(join(bundle, 'config.json')) and isfile(join(bundle, 'vocab.txt')):
            args.bundle = bundle
            args.checkpoint = basename(args.checkpoint)

if args.config and not args.checkpoint and not args.vocab:
    if isdir(args.config):
        # use as bundle path
        args.bundle = args.config
        args.config = None
    elif isfile(args.config):
        # check for bundle
        bundle = dirname(args.config)
        if isfile(join(bundle, 'vocab.txt')) and \
            [f for f in listdir(bundle) if f.endswith('.pth') and isfile(join(bundle, f))]:
            args.bundle = bundle
            args.config = basename(args.config)


if args.bundle:

    print(f'bundle: {args.bundle}')

    if not args.config:
        config = join(args.bundle, 'config.json')
        if isfile(config):
            args.config = config
            print(f'config: <bundle>/{basename(args.config)}')
        else:
            print(f'error: bundle is missing model configuration file config.json')
            sys.exit(1)
    elif not args.config.startswith('/') and not args.config.startswith('./') and not args.config.startswith('../'):
        config = join(args.bundle, args.config)    # relative to bundle
        if isfile(config):
            args.config = config
            print(f'config: <bundle>/{basename(args.config)}')
        elif isfile(args.config):
            print(f'config: {args.config}')
        else:
            print(f'error: missing config: {args.config}')
            sys.exit(1)
    else:
        print(f'config: {args.config}')

    if not args.checkpoint:
        # use first .pth in the bundle
        checkpoint = join(args.bundle, checkpoints[0]) \
                if (checkpoints := [f for f in listdir(args.bundle) if f.endswith('.pth') and isfile(join(args.bundle, f))]) else None
        if checkpoint:
            args.checkpoint = checkpoint
            print(f'chkpt: <bundle>/{basename(args.checkpoint)}')
        else:
            print('error: no checkpoints available at the bundle directory')
            sys.exit(1)
    elif not args.checkpoint.startswith('/') and not args.checkpoint.startswith('./') and not args.checkpoint.startswith('../'):
        checkpoint = join(args.bundle, args.checkpoint)    # relative to bundle
        if isfile(checkpoint):
            args.checkpoint = checkpoint
            print(f'chkpt: <bundle>/{basename(args.checkpoint)}')
        elif isfile(args.checkpoint):
            print(f'chkpt: {args.checkpoint}')
        else:
            print(f'error: missing checkpoint: {args.checkpoint}')
            sys.exit(1)
    else:
        print(f'chkpt: {args.checkpoint}')

    if not args.vocab:
        vocab = join(args.bundle, 'vocab.txt')
        if isfile(vocab):
            args.vocab = vocab
            print(f'vocab: <bundle>/{basename(args.vocab)}')
        else:
            print(f'warning: bundle is missing vocab file vocab.txt')
    elif not args.vocab.startswith('/') and not args.vocab.startswith('./') and not args.vocab.startswith('../'):
        vocab = join(args.bundle, args.vocab)    # relative to bundle
        if isfile(vocab):
            args.vocab = vocab
            print(f'vocab: <bundle>/{basename(args.vocab)}')
        elif isfile(args.vocab):
            print(f'vocab: {args.vocab}')
        else:
            print(f'warning: missing vocab: {args.vocab}')
    else:
        print(f'vocab: {args.vocab}')


if not args.config:
    print('error: model configuration file not specified')
elif not isfile(args.config):
    print(f'error: model configuration file not found: {args.config}')
    args.config = None
if not args.checkpoint:
    print('error: model checkpoint file not specified')
elif not isfile(args.checkpoint):
    print(f'error: model configuration file not found: {args.checkpoint}')
    args.checkpoint = None

if not args.config or not args.checkpoint:
    print('aborting')
    sys.exit(1)

if not args.bundle:
    print(f'config: {args.config}')
    print(f'chkpt: {args.checkpoint}')
    if args.vocab:
        print(f'vocab: {args.vocab}')

if args.speaker is not None:
    print(f'speaker: {args.speaker}')

print(f'text: {args.text}')


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result

if args.vocab:
    with open(args.vocab, 'rt') as f:
        symbols = []
        for symbol in f:
            symbols.append(symbol.rstrip('\r\n'))
    def get_text(text, hps):
        text_norm = []
        for c in text:
            text_norm.append(symbols.index(c))
        if hps.data.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
else:
    from text import symbols, text_to_sequence  # pre-processing input text

    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm


try:
    hps = get_hparams_from_file(args.config)
except Exception as e:
    print('error loading config:', e)
    sys.exit(1)

if args.speaker is not None:
    if hps.data.n_speakers == 0:
        print(f'warning: speaker select, but model is single-speaker, will use the default')
    elif args.speaker > hps.data.n_speakers:
        args.speaker = hps.data.n_speakers
        print(f'warning: speaker id to large, n_speakers = {hps.data.n_speakers}, will use speaker {args.speaker}')

if hps.data.n_speakers > 0:
    print(f'multi-speaker model with {hps.data.n_speakers} speakers')
    if args.speaker is None:
        args.speaker = 0
        print(f'warning: speaker is not set, using speaker {args.speaker}')


if args.cpu:
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    import platform
    if platform.system() == 'Darwin':
        if not torch.backends.mps.is_built():
            print('MPS not available because the current PyTorch install was not built with MPS support.')
        else:
            print('MPS not available because the current MacOS version is not 12.3+ '
                  'and/or you do not have an MPS-enabled device on this machine.')
    elif not torch.backends.cuda.is_built():
        print('CUDA is not available because the current PyTorch install was not built with CUDA support.')

print(f'using device: {device}')


net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)

net_g.eval()

try:
    load_checkpoint(args.checkpoint, net_g, None)
except Exception as e:
    print('error loading checkpoint:', e)
    sys.exit(1)

sid = torch.LongTensor([args.speaker]).to(device) if args.speaker is not None else None
stn_tst = get_text(args.text, hps)

with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    audio = net_g.infer(x_tst, x_tst_lengths, sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    print()
    print(f'generated audio length {len(audio) / hps.data.sampling_rate:.2f}s')

print(f'writing audio to {args.output}')
write(args.output, hps.data.sampling_rate, audio)
