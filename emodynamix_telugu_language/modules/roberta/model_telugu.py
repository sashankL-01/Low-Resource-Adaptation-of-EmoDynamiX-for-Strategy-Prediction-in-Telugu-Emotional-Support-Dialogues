from transformers import XLMRobertaModel, XLMRobertaConfig, XLMRobertaTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch
import json
import numpy as np
from modules.sddp import StructuredDialogueDiscourseParser
from modules.decoder import RobertaClassificationHead
from torch_geometric.nn.conv import RGATConv


class FFN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GATLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_relations, dropout):
        super().__init__()
        self.conv = RGATConv(in_channels=dim_in, out_channels=dim_out, num_relations=num_relations)
        self.ffn = FFN(dim_in=dim_out, dim_hidden=dim_in // 2, dim_out=dim_out, dropout=dropout)

    def forward(self, x, edge_index, edge_type):
        residual = x
        x, attention_weights = self.conv(x, edge_index, edge_type, return_attention_weights=True)
        x = x + residual
        return x, attention_weights

class XLMRBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = XLMRobertaModel.from_pretrained('xlm-roberta-large')
        self.config = XLMRobertaConfig.from_pretrained('xlm-roberta-large')
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        self.max_context_len = 512

    def encode(self, texts):
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_context_len)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = self.encoder(input_ids=tokens['input_ids'].to(self.device), attention_mask=tokens['attention_mask'].to(self.device))
        embeddings = outputs.last_hidden_state
        return embeddings[:, 0, :]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)


class XLMRHeterogeneousGraphTelugu(XLMRBase):
    """Telugu adaptation of heterogeneous graph strategy predictor.

    Differences vs original RobertaHeterogeneousGraph:
    - Uses xlm-roberta-large for context encoding.
    - Integrates external Telugu ERC model (frozen) or accepts precomputed ERC logits / embeddings in samples.
    - Keeps English discourse parser for edge construction.
    - Allows passing 'erc_logits' or 'erc_embeddings' in samples; otherwise computes logits with ERC model.
    """
    def __init__(self, args, lightmode=False):
        super().__init__()
        print(f"[TeluguModel] Device: {self.device}, Lightmode: {lightmode}")
        self.args = args
        self.lightmode = lightmode  # keep parser active unless explicitly disabled
        graph_dim = args.hg_dim
        # context max length for XLM-R
        self.max_context_len = getattr(args, 'context_max_len', 256)

        # Strategy mapping (English original kept)
        if 'esconv' in args.dataset:
            self.strategy2id = json.load(open('data/esconv/strategies.json', 'r', encoding='utf-8'))
        elif 'annomi' in args.dataset:
            self.strategy2id = json.load(open('data/annomi/strategies.json', 'r', encoding='utf-8'))
        self.id2strategy = {v: k for k, v in self.strategy2id.items()}
        self.id2emotion = {0: 'Neutral', 1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Joy', 5: 'Sadness', 6: 'Surprise'}

        # Discourse parser (English pretrained)
        self.dialogue_parser = StructuredDialogueDiscourseParser(
            ckpt_path='pre_trained_models/sddp_stac',
            parse_bs=getattr(args, 'parse_bs', 4096),
            max_contexts_length=getattr(args, 'parse_ctx_len', 48)
        )
        for _, p in self.dialogue_parser.model.named_parameters():
            p.requires_grad = False

        # Telugu ERC model (HuggingFace) path argument
        self.erc_model_path = getattr(args, 'telugu_erc_path', 'telugu_erc_xlmroberta_trained_v2')
        self.erc_model = AutoModelForSequenceClassification.from_pretrained(self.erc_model_path)
        for p in self.erc_model.parameters():
            p.requires_grad = False
        self.erc_hidden_size = self.erc_model.config.hidden_size

        # Emotion prototypes (graph space) & projection for external embeddings
        self.erc_prototypes = nn.Parameter(torch.randn((7, graph_dim)))
        self.erc_proj = nn.Linear(self.erc_hidden_size, graph_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.scalar = 100
        self.t = nn.Parameter(torch.tensor(args.erc_temperature / self.scalar))

        # Graph relations (copied)
        self.graph_relation_dict = {"Continuation": 0, "Question-answer_pair": 1, "Contrast": 2, "Q-Elab": 3,
                                    "Explanation": 4, "Comment": 5, "Background": 6, "Result": 7, "Correction": 8,
                                    "Parallel": 9, "Alternation": 10, "Conditional": 11, "Clarification_question": 12,
                                    "Acknowledgement": 13, "Elaboration": 14, "Narration": 15, "Special": 16,
                                    "Self": 17, "Inter": 18}
        self.graph_relation_dict_inverse = {v: k for k, v in self.graph_relation_dict.items()}
        self.conv1 = GATLayer(graph_dim, graph_dim, len(self.graph_relation_dict.keys()), dropout=0.2)
        self.conv2 = GATLayer(graph_dim, graph_dim, len(self.graph_relation_dict.keys()), dropout=0.2)
        self.conv3 = GATLayer(graph_dim, graph_dim, len(self.graph_relation_dict.keys()), dropout=0.2)
        self.dummy_embedding = nn.Parameter(torch.randn(graph_dim))
        self.strategy_embedding = nn.Embedding(num_embeddings=len(self.strategy2id.keys()), embedding_dim=graph_dim)
        self.node_position_embedding = nn.Embedding(num_embeddings=6, embedding_dim=graph_dim)

        # Classification head
        self.num_classes = len(self.strategy2id) - 1 if args.exclude_others else len(self.strategy2id)
        self.classifier = RobertaClassificationHead(self.config.hidden_size + graph_dim, self.num_classes)

    def _compute_erc_logits(self, texts):
        # Build seeker-only concatenated texts similar to original logic
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=getattr(self.args, 'erc_max_len', 128))
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            outputs = self.erc_model(**tokens)
        logits = outputs.logits  # (B, num_emotions)
        return logits

    def forward(self, samples):
        # Prepare dialogue contexts & ERC input text
        flattened_contexts = []
        dialogues_for_parsing = []
        erc_indices = []
        strategy_indices = []
        dialogue_sizes = []
        utterances_per_dialogue = []
        for i in range(len(samples['dialogue_history'])):
            strategy_history = [int(s.strip()) for s in samples['strategy_history'][i][1:-1].split(',')]
            utterances = samples['dialogue_history'][i].split('</s>')
            speakers = str(samples['speaker_turn'][i]).split(' ')
            dialogue_sizes.append(len(utterances))
            utterances_per_dialogue.append(utterances)
            context = ' '.join([f"[{speakers[j]}] {utterances[j]}" for j in range(len(utterances))])
            dialogue_for_parsing = []
            erc_index = []
            strategy_index = []
            for j in range(len(utterances)):
                turn = {"speaker": speakers[j], "text": utterances[j]}
                dialogue_for_parsing.append(turn)
                if speakers[j] == 'seeker':
                    erc_index.append(j)
                else:
                    strategy = strategy_history[j]
                    strategy = strategy if strategy != -1 else 0
                    strategy_index.append((j, strategy))
            strategy_indices.append(strategy_index)
            erc_indices.append(erc_index)
            dialogues_for_parsing.append(dialogue_for_parsing)
            flattened_contexts.append(context)
        context_embeddings = self.encode(flattened_contexts)

        # Discourse parsing (English model retained)
        if self.lightmode:
            parsed_dialogues = samples['parsed_dialogue']
        else:
            parsed_dialogues = self.dialogue_parser.parse(dialogues_for_parsing)

        # Emotion recognition per-turn alignment
        total_turns = sum(dialogue_sizes)
        if 'erc_logits' in samples:
            erc_logits_full = self.softmax(samples['erc_logits'].to(self.device) / (self.t * self.scalar))
        else:
            seeker_texts = []
            seeker_global_indices = []
            base = 0
            for i in range(len(dialogue_sizes)):
                for j in erc_indices[i]:
                    seeker_texts.append(utterances_per_dialogue[i][j])
                    seeker_global_indices.append(base + j)
                base += dialogue_sizes[i]
            if len(seeker_texts) > 0:
                toks = self.tokenizer(seeker_texts, return_tensors='pt', padding=True, truncation=True, max_length=getattr(self.args, 'erc_max_len', 128))
                toks = {k: v.to(self.device) for k, v in toks.items()}
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    raw_logits = self.erc_model(**toks).logits
                probs = self.softmax(raw_logits / (self.t * self.scalar))
                probs = probs.float()
                erc_logits_full = torch.zeros((total_turns, probs.shape[-1]), device=self.device, dtype=torch.float32)
                erc_logits_full[torch.tensor(seeker_global_indices, device=self.device), :] = probs
            else:
                erc_logits_full = torch.zeros((total_turns, 7), device=self.device)

        if 'erc_embeddings' in samples:
            erc_embeddings_full = samples['erc_embeddings'].to(self.device)
        else:
            if self.args.erc_mixed:
                erc_embeddings_full = erc_logits_full.to(self.erc_prototypes.dtype) @ self.erc_prototypes
            else:
                tags = torch.argmax(erc_logits_full, dim=-1)
                erc_embeddings_full = self.erc_prototypes[tags, :]

        graphs = []
        graph_inputs = {"embeddings": [], "edges": [], "edge_types": []}
        dummy_indices = []
        graph_sizes = []
        for i in range(len(samples['dialogue_history'])):
            nodes = ['DUMMY'] * (dialogue_sizes[i] + 1)
            # Position embeddings are unused and can cause OOB indices; skip lookup
            # pos = torch.tensor([dialogue_sizes[i], ] + np.arange(dialogue_sizes[i]).tolist()).to(self.device)
            # pos_embeddings = self.node_position_embedding(pos)
            node_embeddings = torch.zeros((len(nodes), self.args.hg_dim)).to(self.device)
            node_embeddings[0, :] = node_embeddings[0, :] + self.dummy_embedding
            erc_indices_1 = np.array(erc_indices[i]) + 1
            erc_indices_2 = np.array(erc_indices[i]) + sum(dialogue_sizes[:i])
            if len(erc_indices_1) > 0:
                node_embeddings[erc_indices_1, :] = node_embeddings[erc_indices_1, :] + erc_embeddings_full[erc_indices_2, :]
            strategy_indices_1 = np.array([s[0] for s in strategy_indices[i]]) + 1
            if len(strategy_indices_1) > 0:
                node_embeddings[strategy_indices_1, :] = node_embeddings[strategy_indices_1, :] + self.strategy_embedding(
                    torch.tensor([s[1] for s in strategy_indices[i]]).int().to(self.device))
            # Build node label names
            for j in erc_indices[i]:
                nodes[j + 1] = self.id2emotion[torch.argmax(erc_logits_full[j + sum(dialogue_sizes[:i]), :], dim=-1).item()]
            for j, sid in strategy_indices[i]:
                nodes[j + 1] = self.id2strategy[sid]
            edges = []
            edge_types = []
            for head, tail, tp in parsed_dialogues[i]:
                if head != 0:
                    edges.append([head, tail])
                    edge_types.append(tp)
            for j in range(1, len(nodes)):
                edges.append([j, 0])
                if j - 1 in erc_indices[i]:
                    edge_types.append(self.graph_relation_dict['Inter'])
                else:
                    edge_types.append(self.graph_relation_dict['Self'])
            graph = {"nodes": nodes, "edges": edges, "edge_types": edge_types}
            graphs.append(graph)
            dummy_indices.append(sum(graph_sizes))
            graph_inputs['embeddings'].append(node_embeddings)
            for head, tail in edges:
                graph_inputs['edges'].append([head + sum(graph_sizes), tail + sum(graph_sizes)])
            graph_inputs['edge_types'].extend(edge_types)
            graph_sizes.append(len(nodes))
        graph_inputs['embeddings'] = torch.cat(graph_inputs['embeddings'], dim=0)
        batch_edges = [[], []]
        for head, tail in graph_inputs['edges']:
            batch_edges[0].append(head)
            batch_edges[1].append(tail)
        graph_inputs['edges'] = torch.tensor(batch_edges).to(self.device)
        graph_inputs['edge_types'] = torch.tensor(graph_inputs['edge_types']).to(self.device)

        graph_embeddings, atten_weights_1 = self.conv1(graph_inputs['embeddings'], graph_inputs['edges'], graph_inputs['edge_types'])
        graph_embeddings, atten_weights_2 = self.conv2(graph_embeddings, graph_inputs['edges'], graph_inputs['edge_types'])
        graph_embeddings, atten_weights_3 = self.conv3(graph_embeddings, graph_inputs['edges'], graph_inputs['edge_types'])
        graph_embeddings = graph_embeddings[dummy_indices, :]

        embeddings = torch.cat((graph_embeddings, context_embeddings), dim=-1)
        logits = self.classifier(embeddings)
        return {
            'logits': logits,
            'graphs': graphs,
            'attention_weights': [atten_weights_1, atten_weights_2, atten_weights_3],
            'erc_logits': erc_logits_full
        }
