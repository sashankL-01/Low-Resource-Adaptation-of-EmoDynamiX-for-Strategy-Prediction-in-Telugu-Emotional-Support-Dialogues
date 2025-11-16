import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score


TELUGU_JSON = Path("esconv_telugu/ESConv_Telugu.json")


def iter_dialogues():
    data = json.loads(TELUGU_JSON.read_text(encoding="utf-8"))
    for dlg in data:
        yield dlg


def collect_strategies(split_ratio=(0.8, 0.1, 0.1), seed=114514):
    import random

    all_items = []
    for dlg in iter_dialogues():
        for turn in dlg.get("dialog", []):
            ann = turn.get("annotation") or {}
            strat = ann.get("strategy")
            if strat is not None:
                all_items.append(strat)

    random.Random(seed).shuffle(all_items)
    n = len(all_items)
    n_train = int(split_ratio[0] * n)
    n_valid = int(split_ratio[1] * n)
    train = all_items[:n_train]
    valid = all_items[n_train:n_train + n_valid]
    test = all_items[n_train + n_valid:]
    return train, valid, test


def main():
    train_strats, _, test_strats = collect_strategies()

    y_train = train_strats
    y_test = test_strats

    majority_label, _ = Counter(y_train).most_common(1)[0]
    y_pred = [majority_label] * len(y_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)

    print(json.dumps({
        "baseline": "majority-telugu-dialogue-level",
        "majority_label": majority_label,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "num_train": len(y_train),
        "num_test": len(y_test)
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
