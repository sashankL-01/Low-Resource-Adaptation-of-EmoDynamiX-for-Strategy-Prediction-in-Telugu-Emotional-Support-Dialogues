import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig

from modules.decoder import RobertaClassificationHead


class XLMRDialogueCLSBaseline(nn.Module):
    """Simple baseline: XLM-RoBERTa-large CLS -> MLP for strategy prediction.

    Uses the same label space as ESConv strategies and the same batch contract
    as other models (expects keys from dataloader collate_fn).
    """

    def __init__(self, args):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        self.encoder = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.config = XLMRobertaConfig.from_pretrained("xlm-roberta-large")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")

        if "esconv" in args.dataset:
            strategies_path = Path("data/esconv/strategies.json")
        else:
            raise ValueError(f"Dataset {args.dataset} not supported in XLMRDialogueCLSBaseline")

        self.strategy2id = json.load(open(strategies_path, "r", encoding="utf-8"))
        if getattr(args, "exclude_others", 0):
            self.strategy2id = {k: v for k, v in self.strategy2id.items() if k != "Others"}
        self.id2strategy = {v: k for k, v in self.strategy2id.items()}
        self.num_classes = len(self.strategy2id)

        self.max_context_len = getattr(args, "context_max_len", 256)

        self.classifier = RobertaClassificationHead(
            hidden_size=self.config.hidden_size,
            num_labels=self.num_classes,
        )

    def save(self, path: str):
        """Minimal save interface expected by Trainer.

        Saves a standard state_dict so TrainerForMulticlassClassification
        can reload it with torch.load(..., map_location=...).
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)

    def encode(self, texts):
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = self.encoder(**tokens)
        return out.last_hidden_state[:, 0, :]

    def forward(self, samples):
        contexts = list(samples["dialogue_history"])  # list of strings
        embeddings = self.encode(contexts)
        logits = self.classifier(embeddings)
        return {"logits": logits}
