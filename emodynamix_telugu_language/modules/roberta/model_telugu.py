from transformers import (
    XLMRobertaModel,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    AutoModelForSequenceClassification,
)  # XLM-R encoder + ERC classifier loader
import torch.nn as nn  # Neural network layers
import torch  # PyTorch core
import json  # Strategy mapping files
import numpy as np  # Numeric helpers
from modules.sddp import StructuredDialogueDiscourseParser  # Discourse parser for edges
from modules.decoder import RobertaClassificationHead  # Final classification head
from torch_geometric.nn.conv import RGATConv  # Relational GAT convolution


class FFN(nn.Module):  # Simple 2-layer feed-forward network used inside GAT block
    def __init__(self, dim_in, dim_hidden, dim_out, dropout):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_hidden)  # First projection
        self.relu = nn.ReLU()  # Non-linearity
        self.dropout = nn.Dropout(dropout)  # Regularization
        self.linear2 = nn.Linear(dim_hidden, dim_out)  # Output projection

    def forward(self, x):
        x = self.linear(x)  # (N, dim_hidden)
        x = self.relu(x)  # Activate
        x = self.dropout(x)  # Dropout
        x = self.linear2(x)  # (N, dim_out)
        return x  # Return transformed features


class GATLayer(nn.Module):  # One RGAT layer + residual (optionally FFN)
    def __init__(self, dim_in, dim_out, num_relations, dropout):
        super().__init__()
        self.conv = RGATConv(
            in_channels=dim_in, out_channels=dim_out, num_relations=num_relations
        )  # Relational GAT
        self.ffn = FFN(
            dim_in=dim_out, dim_hidden=dim_in // 2, dim_out=dim_out, dropout=dropout
        )  # Defined but not applied

    def forward(self, x, edge_index, edge_type):
        residual = x  # Save skip connection
        x, attention_weights = self.conv(
            x, edge_index, edge_type, return_attention_weights=True
        )  # RGAT forward
        x = x + residual  # Residual connection for stability
        return x, attention_weights  # Return updated features and attention weights


class XLMRBase(nn.Module):  # Base: tokenizer + XLM-R encoder utilities
    def __init__(self):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Choose GPU if available
        self.encoder = XLMRobertaModel.from_pretrained(
            "xlm-roberta-large"
        )  # XLM-R encoder
        self.config = XLMRobertaConfig.from_pretrained(
            "xlm-roberta-large"
        )  # Config (hidden sizes)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            "xlm-roberta-large"
        )  # Tokenizer
        self.max_context_len = 512  # Default max token length

    def encode(self, texts):  # Encode list[str] -> [CLS]-like embeddings
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
        )  # Batch tokenize
        with torch.amp.autocast(
            "cuda", enabled=torch.cuda.is_available()
        ):  # Mixed precision on CUDA
            outputs = self.encoder(
                input_ids=tokens["input_ids"].to(self.device),
                attention_mask=tokens["attention_mask"].to(self.device),
            )  # Forward
        embeddings = outputs.last_hidden_state  # (B, T, H)
        return embeddings[:, 0, :]  # Use first token representation

    def save(self, path):  # Save all model weights
        torch.save(self.state_dict(), path)

    def load(self, path):  # Load weights (CPU map, non-strict for flexibility)
        self.load_state_dict(
            torch.load(path, map_location=torch.device("cpu")), strict=False
        )


class XLMRHeterogeneousGraphTelugu(
    XLMRBase
):  # Telugu adaptation using XLM-R + RGAT + ERC
    """Telugu adaptation of heterogeneous graph strategy predictor.

    Differences vs original RobertaHeterogeneousGraph:
    - Uses xlm-roberta-large for context encoding.
    - Integrates external Telugu ERC model (frozen) or accepts precomputed ERC logits / embeddings in samples.
    - Keeps English discourse parser for edge construction.
    - Allows passing 'erc_logits' or 'erc_embeddings' in samples; otherwise computes logits with ERC model.
    """

    def __init__(self, args, lightmode=False):
        super().__init__()
        print(
            f"[TeluguModel] Device: {self.device}, Lightmode: {lightmode}"
        )  # Log device + mode
        self.args = args  # Store CLI/Namespace args
        self.lightmode = lightmode  # If True, expect precomputed parses/logits
        graph_dim = args.hg_dim  # Graph feature dimension
        # context max length for XLM-R
        self.max_context_len = getattr(
            args, "context_max_len", 256
        )  # Truncation length for text encoding
        # Optional flag: use English ESConv structure for discourse edges
        self.use_english_structure = getattr(args, "use_english_structure", False)

        # Strategy mapping (English original kept)
        if "esconv" in args.dataset:
            self.strategy2id = json.load(
                open("data/esconv/strategies.json", "r", encoding="utf-8")
            )  # Strategy ids
        elif "annomi" in args.dataset:
            self.strategy2id = json.load(
                open("data/annomi/strategies.json", "r", encoding="utf-8")
            )  # Alt mapping
        self.id2strategy = {
            v: k for k, v in self.strategy2id.items()
        }  # Reverse map: id -> name
        self.id2emotion = {
            0: "Neutral",
            1: "Anger",
            2: "Disgust",
            3: "Fear",
            4: "Joy",
            5: "Sadness",
            6: "Surprise",
        }  # Emotion labels

        # Discourse parser (English pretrained)
        self.dialogue_parser = StructuredDialogueDiscourseParser(
            ckpt_path="pre_trained_models/sddp_stac",  # Path to SDDP checkpoint
            parse_bs=getattr(args, "parse_bs", 4096),  # Parser batch size
            max_contexts_length=getattr(
                args, "parse_ctx_len", 48
            ),  # Max context turns for parser
        )
        for _, p in self.dialogue_parser.model.named_parameters():
            p.requires_grad = False  # Freeze SDDP parameters

        # Load English ESConv dialogues if requested (for structure only)
        if self.use_english_structure:
            try:
                self.english_dialogues = json.load(
                    open("data/esconv/ESConv.json", "r", encoding="utf-8")
                )
            except Exception as e:
                print(
                    f"[TeluguModel] Failed to load English ESConv JSON: {e}. Falling back to Telugu parsing."
                )
                self.use_english_structure = False
            self._english_cache = (
                {}
            )  # Cache parsed English sub-dialogues keyed by (speakers tuple, strategy_history tuple)

        # Telugu ERC model (HuggingFace) path argument
        self.erc_model_path = getattr(
            args, "telugu_erc_path", "telugu_erc_xlmroberta_trained_v2"
        )  # ERC HF path
        self.erc_model = AutoModelForSequenceClassification.from_pretrained(
            self.erc_model_path
        )  # Load ERC head
        for p in self.erc_model.parameters():
            p.requires_grad = False  # Freeze ERC parameters
        self.erc_hidden_size = (
            self.erc_model.config.hidden_size
        )  # Hidden size if using ERC embeddings

        # Emotion prototypes (graph space) & projection for external embeddings
        self.erc_prototypes = nn.Parameter(
            torch.randn((7, graph_dim))
        )  # Learnable emotion class embeddings
        self.erc_proj = nn.Linear(
            self.erc_hidden_size, graph_dim
        )  # Project ERC hidden -> graph dim (not used by default)
        self.softmax = nn.Softmax(dim=-1)  # For logits -> probabilities
        self.scalar = 100  # Scale to keep t small and stable
        self.t = nn.Parameter(
            torch.tensor(args.erc_temperature / self.scalar)
        )  # Learnable temperature (scaled)

        # Graph relations (copied)
        self.graph_relation_dict = {
            "Continuation": 0,
            "Question-answer_pair": 1,
            "Contrast": 2,
            "Q-Elab": 3,
            "Explanation": 4,
            "Comment": 5,
            "Background": 6,
            "Result": 7,
            "Correction": 8,
            "Parallel": 9,
            "Alternation": 10,
            "Conditional": 11,
            "Clarification_question": 12,
            "Acknowledgement": 13,
            "Elaboration": 14,
            "Narration": 15,
            "Special": 16,
            "Self": 17,
            "Inter": 18,
        }  # Relation types used by RGAT
        self.graph_relation_dict_inverse = {
            v: k for k, v in self.graph_relation_dict.items()
        }  # id -> name
        self.conv1 = GATLayer(
            graph_dim, graph_dim, len(self.graph_relation_dict.keys()), dropout=0.2
        )  # RGAT layer 1
        self.conv2 = GATLayer(
            graph_dim, graph_dim, len(self.graph_relation_dict.keys()), dropout=0.2
        )  # RGAT layer 2
        self.conv3 = GATLayer(
            graph_dim, graph_dim, len(self.graph_relation_dict.keys()), dropout=0.2
        )  # RGAT layer 3
        self.dummy_embedding = nn.Parameter(
            torch.randn(graph_dim)
        )  # Global node embedding per dialogue
        self.strategy_embedding = nn.Embedding(
            num_embeddings=len(self.strategy2id.keys()), embedding_dim=graph_dim
        )  # Supporter strategy ids -> emb
        self.node_position_embedding = nn.Embedding(
            num_embeddings=6, embedding_dim=graph_dim
        )  # Positional emb (disabled below)

        # Classification head
        self.num_classes = (
            len(self.strategy2id) - 1 if args.exclude_others else len(self.strategy2id)
        )  # Optionally drop 'Others'
        self.classifier = RobertaClassificationHead(
            self.config.hidden_size + graph_dim, self.num_classes
        )  # Final MLP

    def _parse_english_subdialogue(self, speakers, strategy_history):
        if not self.use_english_structure:
            return None
        key = (tuple(speakers), tuple(strategy_history))
        if key in self._english_cache:
            return self._english_cache[key]
        # Iterate English dialogues
        for d in self.english_dialogues:
            eng_speakers = []
            eng_strategy_hist = []
            turns_prefix = []
            for turn in d.get("dialog", []):
                spk = turn.get("speaker", "")
                # Normalize to seeker/supporter
                spk_norm = "seeker" if spk == "seeker" else "supporter"
                eng_speakers.append(spk_norm)
                ann = turn.get("annotation", {}) or {}
                if spk_norm == "supporter":
                    strat_name = ann.get("strategy", "Others")
                    if strat_name not in self.strategy2id:
                        strat_name = "Others"
                    strat_id = self.strategy2id.get(strat_name, 0)
                    eng_strategy_hist.append(strat_id)
                else:
                    eng_strategy_hist.append(-1)
                turns_prefix.append(
                    {"speaker": spk_norm, "text": turn.get("content", "")}
                )
                # Once we reach desired length, check match
                if len(turns_prefix) == len(speakers):
                    if (
                        eng_speakers == speakers
                        and eng_strategy_hist == strategy_history
                    ):
                        try:
                            parsed = self.dialogue_parser.parse([turns_prefix])[0]
                        except Exception as e:
                            print(f"[TeluguModel] English parse failed: {e}")
                            parsed = None
                        self._english_cache[key] = parsed
                        return parsed
                    break  # Length reached but not matching; move to next dialogue
        # Not found
        self._english_cache[key] = None
        return None

    def _compute_erc_logits(self, texts):  # Helper to run ERC model over texts
        # Build seeker-only concatenated texts similar to original logic
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.args, "erc_max_len", 128),
        )  # Tokenize
        tokens = {k: v.to(self.device) for k, v in tokens.items()}  # Move to device
        with torch.amp.autocast(
            "cuda", enabled=torch.cuda.is_available()
        ):  # Mixed precision on CUDA
            outputs = self.erc_model(**tokens)  # Forward ERC classifier
        logits = outputs.logits  # (B, num_emotions)
        return logits  # Return raw logits

    def forward(self, samples):  # Main forward over a batch from dataloader
        # Prepare dialogue contexts & ERC input text
        flattened_contexts = []  # Text for XLM-R encoding (one per sample)
        dialogues_for_parsing = []  # Turn-structured inputs to SDDP
        erc_indices = []  # Positions of seeker turns (for ERC)
        strategy_indices = []  # (turn_idx, strategy_id) for supporter turns
        dialogue_sizes = []  # #turns per dialogue
        utterances_per_dialogue = []  # Cache utterances per sample
        for i in range(len(samples["dialogue_history"])):  # Iterate batch samples
            strategy_history = [
                int(s.strip()) for s in samples["strategy_history"][i][1:-1].split(",")
            ]  # Parse list string -> ints
            utterances = samples["dialogue_history"][i].split(
                "</s>"
            )  # Split utterances
            speakers = str(samples["speaker_turn"][i]).split(
                " "
            )  # Speakers aligned with turns
            dialogue_sizes.append(len(utterances))  # Record length
            utterances_per_dialogue.append(utterances)  # Keep for ERC batching
            context = " ".join(
                [f"[{speakers[j]}] {utterances[j]}" for j in range(len(utterances))]
            )  # Build context text
            dialogue_for_parsing = []  # List[{'speaker','text'}]
            erc_index = []  # Seeker turn positions
            strategy_index = []  # Supporter positions + strategy id
            for j in range(len(utterances)):
                turn = {"speaker": speakers[j], "text": utterances[j]}  # One turn
                dialogue_for_parsing.append(turn)  # For SDDP
                if speakers[j] == "seeker":  # Seeker turns -> ERC
                    erc_index.append(j)
                else:  # Supporter turns -> strategy ids
                    strategy = strategy_history[j]
                    strategy = (
                        strategy if strategy != -1 else 0
                    )  # Replace -1 placeholder with 0
                    strategy_index.append((j, strategy))
            strategy_indices.append(strategy_index)  # Collect per dialogue
            erc_indices.append(erc_index)  # Collect per dialogue
            dialogues_for_parsing.append(dialogue_for_parsing)  # For parser
            flattened_contexts.append(context)  # For XLM-R
        context_embeddings = self.encode(flattened_contexts)  # (B, H_enc)

        # Discourse parsing:
        # Priority order:
        # 1. lightmode -> use provided parsed_dialogue
        # 2. use_english_structure -> attempt English prefix match & parse English
        # 3. fallback -> parse Telugu turns directly
        if self.lightmode:
            parsed_dialogues = samples["parsed_dialogue"]  # Provided externally
        else:
            parsed_dialogues = []
            for i in range(len(dialogues_for_parsing)):
                speakers_i = [t["speaker"] for t in dialogues_for_parsing[i]]
                # Reconstruct strategy history up to this sample (already parsed earlier)
                strategy_history = [
                    int(s.strip())
                    for s in samples["strategy_history"][i][1:-1].split(",")
                ]
                english_parsed = (
                    self._parse_english_subdialogue(speakers_i, strategy_history)
                    if self.use_english_structure
                    else None
                )
                if english_parsed is not None:
                    parsed_dialogues.append(english_parsed)
                else:
                    # Fallback: parse current (Telugu) turns
                    parsed_dialogues.append(
                        self.dialogue_parser.parse([dialogues_for_parsing[i]])[0]
                    )

        # Emotion recognition per-turn alignment
        total_turns = sum(dialogue_sizes)  # Total nodes (excluding dummy) across batch
        if "erc_logits" in samples:  # Precomputed ERC provided
            erc_logits_full = self.softmax(
                samples["erc_logits"].to(self.device) / (self.t * self.scalar)
            )  # Temperature-softmax
        else:
            seeker_texts = []  # Texts of seeker turns (batched across dialogues)
            seeker_global_indices = []  # Their positions in concatenated turns
            base = 0  # Running base offset per dialogue
            for i in range(len(dialogue_sizes)):
                for j in erc_indices[i]:
                    seeker_texts.append(
                        utterances_per_dialogue[i][j]
                    )  # Append seeker utterance
                    seeker_global_indices.append(base + j)  # Record global index
                base += dialogue_sizes[i]  # Shift offset
            if len(seeker_texts) > 0:
                toks = self.tokenizer(
                    seeker_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.args, "erc_max_len", 128),
                )  # Tokenize seeker texts
                toks = {k: v.to(self.device) for k, v in toks.items()}  # To device
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    raw_logits = self.erc_model(**toks).logits  # Run ERC classifier
                probs = self.softmax(
                    raw_logits / (self.t * self.scalar)
                )  # Temperature-softmax
                probs = probs.float()  # Ensure float32
                erc_logits_full = torch.zeros(
                    (total_turns, probs.shape[-1]),
                    device=self.device,
                    dtype=torch.float32,
                )  # Allocate per-turn table
                erc_logits_full[
                    torch.tensor(seeker_global_indices, device=self.device), :
                ] = probs  # Fill seeker rows
            else:
                erc_logits_full = torch.zeros(
                    (total_turns, 7), device=self.device
                )  # No seeker turns -> zeros

        if "erc_embeddings" in samples:  # If graph-space ERC embeddings provided
            erc_embeddings_full = samples["erc_embeddings"].to(self.device)
        else:
            if self.args.erc_mixed:  # Mixed: probs @ prototypes -> dense embedding
                erc_embeddings_full = (
                    erc_logits_full.to(self.erc_prototypes.dtype) @ self.erc_prototypes
                )
            else:  # Hard tag to prototype
                tags = torch.argmax(erc_logits_full, dim=-1)
                erc_embeddings_full = self.erc_prototypes[tags, :]

        graphs = []  # For case study / visualization
        graph_inputs = {
            "embeddings": [],
            "edges": [],
            "edge_types": [],
        }  # Accumulate batched graph
        dummy_indices = []  # Indices of dummy nodes after concatenation
        graph_sizes = []  # Number of nodes (incl. dummy) per dialogue
        for i in range(len(samples["dialogue_history"])):
            nodes = ["DUMMY"] * (dialogue_sizes[i] + 1)  # 1 dummy + one per turn
            # Position embeddings are unused and can cause OOB indices; skip lookup  # Disabled to avoid index errors
            # pos = torch.tensor([dialogue_sizes[i], ] + np.arange(dialogue_sizes[i]).tolist()).to(self.device)
            # pos_embeddings = self.node_position_embedding(pos)
            node_embeddings = torch.zeros((len(nodes), self.args.hg_dim)).to(
                self.device
            )  # Initialize node features
            node_embeddings[0, :] = (
                node_embeddings[0, :] + self.dummy_embedding
            )  # Dummy feature
            erc_indices_1 = np.array(erc_indices[i]) + 1  # Shift for dummy at index 0
            erc_indices_2 = np.array(erc_indices[i]) + sum(
                dialogue_sizes[:i]
            )  # Global seeker indices
            if len(erc_indices_1) > 0:
                node_embeddings[erc_indices_1, :] = (
                    node_embeddings[erc_indices_1, :]
                    + erc_embeddings_full[erc_indices_2, :]
                )  # Add ERC embeddings to seeker nodes
            strategy_indices_1 = (
                np.array([s[0] for s in strategy_indices[i]]) + 1
            )  # Supporter local indices (+1)
            if len(strategy_indices_1) > 0:
                node_embeddings[strategy_indices_1, :] = node_embeddings[
                    strategy_indices_1, :
                ] + self.strategy_embedding(
                    torch.tensor([s[1] for s in strategy_indices[i]])
                    .int()
                    .to(self.device)
                )  # Add strategy emb
            # Build node label names (for visualization only)
            for j in erc_indices[i]:
                nodes[j + 1] = self.id2emotion[
                    torch.argmax(
                        erc_logits_full[j + sum(dialogue_sizes[:i]), :], dim=-1
                    ).item()
                ]  # Seeker node label
            for j, sid in strategy_indices[i]:
                nodes[j + 1] = self.id2strategy[sid]  # Supporter node label
            edges = []  # Edge list (head, tail)
            edge_types = []  # Relation ids per edge
            for head, tail, tp in parsed_dialogues[i]:  # Discourse edges
                if head != 0:  # Skip potential root marker
                    edges.append([head, tail])
                    edge_types.append(tp)
            for j in range(1, len(nodes)):  # Star edges to dummy for all nodes
                edges.append([j, 0])
                if j - 1 in erc_indices[i]:
                    edge_types.append(
                        self.graph_relation_dict["Inter"]
                    )  # Seeker -> dummy uses Inter
                else:
                    edge_types.append(
                        self.graph_relation_dict["Self"]
                    )  # Others -> dummy uses Self
            graph = {
                "nodes": nodes,
                "edges": edges,
                "edge_types": edge_types,
            }  # Save per-dialogue graph
            graphs.append(graph)  # Collect
            dummy_indices.append(
                sum(graph_sizes)
            )  # Dummy index after concatenation offset
            graph_inputs["embeddings"].append(
                node_embeddings
            )  # Accumulate node features
            for head, tail in edges:  # Shift edges by previous graphs' sizes
                graph_inputs["edges"].append(
                    [head + sum(graph_sizes), tail + sum(graph_sizes)]
                )
            graph_inputs["edge_types"].extend(edge_types)  # Accumulate relation ids
            graph_sizes.append(len(nodes))  # Track size for next offsets
        graph_inputs["embeddings"] = torch.cat(
            graph_inputs["embeddings"], dim=0
        )  # (sum_N, hg_dim)
        batch_edges = [[], []]  # COO format for PyG
        for head, tail in graph_inputs["edges"]:
            batch_edges[0].append(head)
            batch_edges[1].append(tail)
        graph_inputs["edges"] = torch.tensor(batch_edges).to(self.device)  # 2 x E
        graph_inputs["edge_types"] = torch.tensor(graph_inputs["edge_types"]).to(
            self.device
        )  # (E,)

        graph_embeddings, atten_weights_1 = self.conv1(
            graph_inputs["embeddings"],
            graph_inputs["edges"],
            graph_inputs["edge_types"],
        )  # RGAT 1
        graph_embeddings, atten_weights_2 = self.conv2(
            graph_embeddings, graph_inputs["edges"], graph_inputs["edge_types"]
        )  # RGAT 2
        graph_embeddings, atten_weights_3 = self.conv3(
            graph_embeddings, graph_inputs["edges"], graph_inputs["edge_types"]
        )  # RGAT 3
        graph_embeddings = graph_embeddings[
            dummy_indices, :
        ]  # Pool per-dialogue using dummy node embedding

        embeddings = torch.cat(
            (graph_embeddings, context_embeddings), dim=-1
        )  # Concatenate graph + text features
        logits = self.classifier(embeddings)  # Predict next strategy
        return {
            "logits": logits,  # (B, num_classes)
            "graphs": graphs,  # For inspection/visualization
            "attention_weights": [
                atten_weights_1,
                atten_weights_2,
                atten_weights_3,
            ],  # RGAT attentions
            "erc_logits": erc_logits_full,  # ERC probabilities aligned to turns
        }
