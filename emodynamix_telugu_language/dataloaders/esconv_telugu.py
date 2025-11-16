import json
import torch
from torch.utils.data import Dataset
import random

class ESConvTelugu(Dataset):
    """Dataset builder for Telugu translated ESConv (ESConv_Telugu.json).

    We generate training samples by predicting the next supporter strategy
    at each supporter turn except the final supporter turn in a dialogue.

    Each sample contains:
    - dialogue_history: utterances joined by '</s>'
    - strategy_history: list of supporter strategies aligned with speaker turns (int ids, -1 for seeker turns)
    - speaker_turn: space separated speaker sequence ("seeker"/"supporter")
    - gold_standard: placeholder (unused)
    - label: next supporter strategy id
    - feedback: aggregated feedback score (fallback 5)

    Discourse parsing & ERC handled inside model; we do not precompute edges or emotion logits here.
    """
    def __init__(self, split: str, args):
        super().__init__()
        assert split in ["train", "valid", "test"], "split must be train|valid|test"
        self.args = args
        # Strategy mapping reused from original English file
        self.strategy2id = json.load(open('data/esconv/strategies.json', 'r', encoding='utf-8'))
        if args.exclude_others:
            self.strategy2id = {k: v for k, v in self.strategy2id.items() if k != 'Others'}
        self.id2strategy = {v: k for k, v in self.strategy2id.items()}
        self.id2label = self.id2strategy
        # Load full Telugu dialogues
        data = json.load(open('esconv_telugu/ESConv_Telugu.json', 'r', encoding='utf-8'))
        random.seed(args.seed)
        random.shuffle(data)
        n = len(data)
        n_train = int(0.8 * n)
        n_valid = int(0.1 * n)
        train_data = data[:n_train]
        valid_data = data[n_train:n_train + n_valid]
        test_data = data[n_train + n_valid:]
        if split == 'train':
            raw_dialogues = train_data
        elif split == 'valid':
            raw_dialogues = valid_data
        else:
            raw_dialogues = test_data
        self.samples = []
        class_counts = [0] * len(self.strategy2id.keys())
        for d in raw_dialogues:
            utterances = []
            speaker_seq = []
            strategy_history = []  # aligned with turns: strategy id for supporter, -1 for seeker
            feedback_scores = []
            for turn in d['dialog']:
                speaker = turn['speaker']
                text = turn['content']
                utterances.append(text)
                speaker_seq.append(speaker)
                ann = turn.get('annotation', {}) or {}
                if speaker == 'supporter':
                    strategy_name = ann.get('strategy', 'Others')
                    # normalize unexpected strategy names
                    if strategy_name not in self.strategy2id:
                        strategy_name = 'Others'
                    strategy_id = self.strategy2id[strategy_name]
                    strategy_history.append(strategy_id)
                else:
                    strategy_history.append(-1)
                if speaker == 'seeker':
                    fb = ann.get('feedback')
                    if fb is not None:
                        try:
                            feedback_scores.append(int(fb))
                        except Exception:
                            pass
            # Build prediction samples: each supporter turn (index t) predicts next supporter strategy.
            supporter_turn_indices = [i for i, s in enumerate(speaker_seq) if s == 'supporter']
            for si, turn_idx in enumerate(supporter_turn_indices):
                if si == len(supporter_turn_indices) - 1:
                    continue  # no next supporter turn -> skip
                next_turn_idx = supporter_turn_indices[si + 1]
                next_strategy_id = strategy_history[next_turn_idx]
                if self.args.exclude_others and next_strategy_id == self.strategy2id.get('Others', -999):
                    continue
                feedback = max(feedback_scores) if feedback_scores else 5
                # Build strategy history string representation
                strategy_hist_str = '[' + ','.join(str(x) for x in strategy_history[:turn_idx + 1]) + ']'
                dialogue_hist_str = '</s>'.join(utterances[:turn_idx + 1])
                speaker_turn_str = ' '.join(speaker_seq[:turn_idx + 1])
                sample = {
                    'dialogue_history': dialogue_hist_str,
                    'strategy_history': strategy_hist_str,
                    'speaker_turn': speaker_turn_str,
                    'gold_standard': '',
                    'label': next_strategy_id,
                    'feedback': feedback,
                }
                self.samples.append(sample)
                class_counts[next_strategy_id] += 1
        # Class weights
        self.class_weights = [sum(class_counts) / len(class_counts) / c if c > 0 else 0.0 for c in class_counts]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, batch):
        return {
            'dialogue_history': [b['dialogue_history'] for b in batch],
            'strategy_history': [b['strategy_history'] for b in batch],
            'speaker_turn': [b['speaker_turn'] for b in batch],
            'gold_standard': [b['gold_standard'] for b in batch],
            'label': torch.tensor([b['label'] for b in batch]),
            'feedback': [b['feedback'] for b in batch],
        }
