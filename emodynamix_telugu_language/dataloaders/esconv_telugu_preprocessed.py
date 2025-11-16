import torch
from torch.utils.data import Dataset
import pickle
import os

class ESConvTeluguPreProcessed(Dataset):
    """Loads preprocessed Telugu ESConv samples with parsed_dialogue & optional erc logits.
    Expected pickle path: esconv_telugu_preprocessed/{split}.pkl
    Each element: {
        'dialogue_history': str,
        'strategy_history': str,
        'speaker_turn': str,
        'gold_standard': str,
        'label': int,
        'feedback': int,
        'parsed_dialogue': list[(head, tail, relation_id)],
        Optional: 'erc_logits': tensor(shape=(num_seeker_turns, 7))
    }
    """
    def __init__(self, split, args):
        super().__init__()
        assert split in ['train', 'valid', 'test']
        self.args = args
        self.strategy2id = pickle.load(open('esconv_telugu_preprocessed/strategy2id.pkl', 'rb')) if os.path.exists('esconv_telugu_preprocessed/strategy2id.pkl') else None
        if self.strategy2id is None:
            # Fallback to English mapping
            import json
            self.strategy2id = json.load(open('data/esconv/strategies.json', 'r', encoding='utf-8'))
        if args.exclude_others:
            self.strategy2id = {k: v for k, v in self.strategy2id.items() if k != 'Others'}
        self.id2label = {v: k for k, v in self.strategy2id.items()}
        pkl_path = f'esconv_telugu_preprocessed/{split}.pkl'
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Preprocessed file {pkl_path} not found. Run make_telugu_preprocessed.py first.")
        self.data = pickle.load(open(pkl_path, 'rb'))
        # class weights
        class_counts = [0] * len(self.strategy2id)
        for d in self.data:
            class_counts[d['label']] += 1
        self.class_weights = [sum(class_counts) / len(class_counts) / c if c > 0 else 0 for c in class_counts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        out = {
            'dialogue_history': [b['dialogue_history'] for b in batch],
            'strategy_history': [b['strategy_history'] for b in batch],
            'speaker_turn': [b['speaker_turn'] for b in batch],
            'gold_standard': [b['gold_standard'] for b in batch],
            'label': torch.tensor([b['label'] for b in batch]),
            'feedback': [b['feedback'] for b in batch],
            'parsed_dialogue': [b['parsed_dialogue'] for b in batch],
        }
        # Concatenate erc logits if present
        if 'erc_logits' in batch[0]:
            out['erc_logits'] = torch.cat([b['erc_logits'] for b in batch], dim=0)
        return out
