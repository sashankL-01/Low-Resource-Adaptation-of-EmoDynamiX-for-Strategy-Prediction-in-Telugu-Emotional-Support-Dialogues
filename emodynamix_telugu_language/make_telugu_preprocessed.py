"""Precompute parsed_dialogue and ERC logits for Telugu ESConv to accelerate training.

Outputs directory: esconv_telugu_preprocessed/
Files: train.pkl, valid.pkl, test.pkl, strategy2id.pkl

Run:
    python make_telugu_preprocessed.py --telugu_erc_path telugu_erc_xlmroberta_trained_v2 --hg_dim 512

Then train with:
    python main.py --mode train --model xlmr-hg-telugu --dataset esconv-telugu-preprocessed --lightmode 1 --hg_dim 512 --telugu_erc_path telugu_erc_xlmroberta_trained_v2
"""
import argparse
import json
import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from modules.sddp import StructuredDialogueDiscourseParser

def build_samples(full_dialogues, strategy2id, erc_model, tokenizer):
    samples = []
    for d in full_dialogues:
        utterances = []
        speakers = []
        strategy_history = []
        feedback_scores = []
        for turn in d['dialog']:
            utterances.append(turn['content'])
            speakers.append(turn['speaker'])
            ann = turn.get('annotation', {}) or {}
            if turn['speaker'] == 'supporter':
                strategy_name = ann.get('strategy', 'Others')
                if strategy_name not in strategy2id:
                    strategy_name = 'Others'
                strategy_history.append(strategy2id[strategy_name])
            else:
                strategy_history.append(-1)
            if turn['speaker'] == 'seeker' and 'feedback' in ann:
                try:
                    feedback_scores.append(int(ann['feedback']))
                except Exception:
                    pass
        supporter_turn_indices = [i for i, s in enumerate(speakers) if s == 'supporter']
        for si, turn_idx in enumerate(supporter_turn_indices):
            if si == len(supporter_turn_indices) - 1:
                continue
            next_turn_idx = supporter_turn_indices[si + 1]
            next_strategy_id = strategy_history[next_turn_idx]
            feedback = max(feedback_scores) if feedback_scores else 5
            strategy_hist_str = '[' + ','.join(str(x) for x in strategy_history[:turn_idx + 1]) + ']'
            dialogue_hist_str = '</s>'.join(utterances[:turn_idx + 1])
            speaker_turn_str = ' '.join(speakers[:turn_idx + 1])
            # ERC logits over concatenated seeker text so far
            seeker_text = ' '.join([utterances[j] for j in range(turn_idx + 1) if speakers[j] == 'seeker'])
            if seeker_text.strip():
                toks = tokenizer(seeker_text, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    logits = erc_model(**{k: v.to(erc_model.device) for k, v in toks.items()}).logits.cpu()
            else:
                logits = torch.zeros((1, 7))
            sample = {
                'dialogue_history': dialogue_hist_str,
                'strategy_history': strategy_hist_str,
                'speaker_turn': speaker_turn_str,
                'gold_standard': '',
                'label': next_strategy_id,
                'feedback': feedback,
                'erc_logits': logits,
                'raw_dialogue_turns': [{'speaker': speakers[j], 'text': utterances[j]} for j in range(turn_idx + 1)],
            }
            samples.append(sample)
    return samples

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--telugu_erc_path', type=str, default='telugu_erc_xlmroberta_trained_v2')
    ap.add_argument('--seed', type=int, default=114514)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    strategy2id = json.load(open('data/esconv/strategies.json', 'r', encoding='utf-8'))
    full = json.load(open('esconv_telugu/ESConv_Telugu.json', 'r', encoding='utf-8'))
    n = len(full)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    train_data = full[:n_train]
    valid_data = full[n_train:n_train + n_valid]
    test_data = full[n_train + n_valid:]

    erc_model = AutoModelForSequenceClassification.from_pretrained(args.telugu_erc_path)
    erc_model.eval()
    erc_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    tokenizer = AutoTokenizer.from_pretrained(args.telugu_erc_path)

    print('Building raw samples with ERC logits...')
    train_samples = build_samples(train_data, strategy2id, erc_model, tokenizer)
    valid_samples = build_samples(valid_data, strategy2id, erc_model, tokenizer)
    test_samples = build_samples(test_data, strategy2id, erc_model, tokenizer)

    print('Parsing dialogues (English discourse parser)...')
    parser = StructuredDialogueDiscourseParser(ckpt_path='pre_trained_models/sddp_stac', parse_bs=4096)
    for s in train_samples + valid_samples + test_samples:
        parsed = parser.parse([{'speaker': t['speaker'], 'text': t['text']} for t in s['raw_dialogue_turns']])
        s['parsed_dialogue'] = parsed
        del s['raw_dialogue_turns']

    os.makedirs('esconv_telugu_preprocessed', exist_ok=True)
    pickle.dump(train_samples, open('esconv_telugu_preprocessed/train.pkl', 'wb'))
    pickle.dump(valid_samples, open('esconv_telugu_preprocessed/valid.pkl', 'wb'))
    pickle.dump(test_samples, open('esconv_telugu_preprocessed/test.pkl', 'wb'))
    pickle.dump(strategy2id, open('esconv_telugu_preprocessed/strategy2id.pkl', 'wb'))
    print('Done: saved preprocessed Telugu datasets.')

if __name__ == '__main__':
    main()
