"""
Comprehensive evaluation script for EmoDynamiX-Telugu.

Performs:
1. Main evaluation on test set with all metrics
2. Confusion matrix visualization
3. Ablation study (w/o graph, w/o emotion, w/o discourse, w/o dummy)
4. Results saved to JSON and figures
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders import ESConvTelugu
from metrics import preference_bias
from modules.roberta.model_telugu import XLMRHeterogeneousGraphTelugu
from utils import seed_everything


def compute_all_metrics(
    predictions: np.ndarray,
    truths: np.ndarray,
    id2label: Dict[int, str]
) -> Dict:
    """Compute all evaluation metrics matching EmoDynamiX paper."""
    acc = accuracy_score(truths, predictions)
    macro_f1 = f1_score(truths, predictions, average='macro')
    micro_f1 = f1_score(truths, predictions, average='micro')
    weighted_f1 = f1_score(truths, predictions, average='weighted')
    
    per_class_f1 = f1_score(truths, predictions, average=None)
    c_matrix = confusion_matrix(truths, predictions)
    pref_bias = preference_bias(c_matrix)
    
    metrics = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'weighted_f1': float(weighted_f1),
        'preference_bias': float(pref_bias),
        'confusion_matrix': c_matrix.tolist(),
    }
    
    # Per-class F1 scores
    for _id in range(len(id2label)):
        if _id < len(per_class_f1):
            metrics[id2label[_id]] = float(per_class_f1[_id])
        else:
            metrics[id2label[_id]] = 0.0
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str]
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Run inference and compute metrics."""
    model.eval()
    predictions = []
    truths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(batch)
            y_pred = outputs["logits"]
            predictions.append(torch.argmax(y_pred, dim=-1))
            truths.append(batch["label"])
    
    predictions = torch.cat(predictions, dim=-1).int().cpu().numpy()
    truths = torch.cat(truths, dim=-1).cpu().numpy()
    
    metrics = compute_all_metrics(predictions, truths, id2label)
    return metrics, predictions, truths


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: str,
    normalize: bool = True
):
    """Generate and save confusion matrix heatmap."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.xlabel('Predicted Strategy', fontsize=12)
    plt.ylabel('True Strategy', fontsize=12)
    plt.title('Confusion Matrix - EmoDynamiX-Telugu', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


class AblationModel(torch.nn.Module):
    """Wrapper to run ablations by modifying forward pass."""
    
    def __init__(self, base_model: XLMRHeterogeneousGraphTelugu, ablation_type: str):
        super().__init__()
        self.base_model = base_model
        self.ablation_type = ablation_type
        
    def forward(self, samples):
        if self.ablation_type == "no_graph":
            return self._forward_no_graph(samples)
        elif self.ablation_type == "no_emotion":
            return self._forward_no_emotion(samples)
        elif self.ablation_type == "no_discourse":
            return self._forward_no_discourse(samples)
        elif self.ablation_type == "no_dummy":
            return self._forward_no_dummy(samples)
        else:
            return self.base_model(samples)
    
    def _forward_no_graph(self, samples):
        """Skip RGAT layers, use only XLM-R context embeddings."""
        # Build contexts from dialogue history
        flattened_contexts = []
        for i in range(len(samples['dialogue_history'])):
            context_str = samples['dialogue_history'][i]
            flattened_contexts.append(context_str)
        
        # Encode with XLM-R
        context_embeddings = self.base_model.encode(flattened_contexts)
        
        # Direct classification without graph
        logits = self.base_model.classifier(context_embeddings)
        
        return {
            "logits": logits,
            "erc_logits": torch.zeros(1, 7, device=logits.device),
            "graphs": [],
            "attention_weights": []
        }
    
    def _forward_no_emotion(self, samples):
        """Remove ERC embeddings from graph by zeroing them out."""
        # Temporarily override ERC prototypes with zeros
        original_prototypes = self.base_model.erc_prototypes.clone()
        self.base_model.erc_prototypes = torch.zeros_like(original_prototypes)
        
        result = self.base_model(samples)
        
        # Restore original
        self.base_model.erc_prototypes = original_prototypes
        return result
    
    def _forward_no_discourse(self, samples):
        """Replace discourse edges with sequential edges only."""
        # Build simple sequential edges for each dialogue
        if 'parsed_dialogue' not in samples or samples['parsed_dialogue'] is None:
            # Need to compute dialogue sizes
            dialogue_sizes = []
            for i in range(len(samples['dialogue_history'])):
                utterances = samples['dialogue_history'][i].split('</s>')
                utterances = [u.strip() for u in utterances if u.strip()]
                dialogue_sizes.append(len(utterances))
            
            sequential_parsed = []
            for size in dialogue_sizes:
                # Simple sequential: turn i -> turn i+1
                edges = [(i, i + 1, "sequential") for i in range(size - 1)]
                sequential_parsed.append(edges)
            
            samples_copy = samples.copy()
            samples_copy['parsed_dialogue'] = sequential_parsed
        else:
            # Replace existing parsed with sequential
            samples_copy = samples.copy()
            sequential_parsed = []
            for parsed in samples['parsed_dialogue']:
                # Count nodes from existing parse
                max_node = max([max(e[0], e[1]) for e in parsed] + [0])
                edges = [(i, i + 1, "sequential") for i in range(max_node)]
                sequential_parsed.append(edges)
            samples_copy['parsed_dialogue'] = sequential_parsed
        
        return self.base_model(samples_copy)
    
    def _forward_no_dummy(self, samples):
        """Remove dummy node by zeroing its embedding."""
        original_dummy = self.base_model.dummy_embedding.clone()
        self.base_model.dummy_embedding = torch.nn.Parameter(
            torch.zeros_like(original_dummy)
        )
        
        result = self.base_model(samples)
        
        self.base_model.dummy_embedding = torch.nn.Parameter(original_dummy)
        return result


def run_ablation_study(
    model: XLMRHeterogeneousGraphTelugu,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    baseline_metrics: Dict
) -> List[Dict]:
    """Run all ablation experiments."""
    ablations = [
        ("w/o Graph Learning", "no_graph"),
        ("w/o Mixed Emotion", "no_emotion"),
        ("w/o Discourse Parser", "no_discourse"),
        ("w/o Dummy Node", "no_dummy"),
    ]
    
    results = []
    baseline_macro = baseline_metrics['macro_f1']
    
    for variant_name, ablation_type in ablations:
        print(f"\n{'='*60}")
        print(f"Running ablation: {variant_name}")
        print(f"{'='*60}")
        
        ablation_model = AblationModel(model, ablation_type)
        ablation_model.to(device)
        
        metrics, _, _ = evaluate_model(ablation_model, dataloader, device, id2label)
        
        result = {
            "variant": variant_name,
            "macro_f1": metrics['macro_f1'],
            "delta_macro_f1": baseline_macro - metrics['macro_f1'],
            "weighted_f1": metrics['weighted_f1'],
            "accuracy": metrics['accuracy'],
            "preference_bias": metrics['preference_bias']
        }
        results.append(result)
        
        print(f"  Macro F1: {metrics['macro_f1']:.4f} (Δ = {result['delta_macro_f1']:+.4f})")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Preference Bias: {metrics['preference_bias']:.4f}")
    
    return results


def print_results_table(baseline_metrics: Dict, ablation_results: List[Dict]):
    """Print results in EmoDynamiX Table 2 format."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS - EmoDynamiX-Telugu")
    print("="*80)
    
    print("\n1. Main Metrics (Full Model)")
    print("-" * 80)
    print(f"{'Metric':<25} {'Value':>10}")
    print("-" * 80)
    print(f"{'Accuracy':<25} {baseline_metrics['accuracy']:>10.4f}")
    print(f"{'Macro F1':<25} {baseline_metrics['macro_f1']:>10.4f}")
    print(f"{'Micro F1':<25} {baseline_metrics['micro_f1']:>10.4f}")
    print(f"{'Weighted F1':<25} {baseline_metrics['weighted_f1']:>10.4f}")
    print(f"{'Preference Bias':<25} {baseline_metrics['preference_bias']:>10.4f}")
    
    print("\n2. Per-Class F1 Scores")
    print("-" * 80)
    strategies = [
        "Question", "Restatement or Paraphrasing", "Reflection of feelings",
        "Self-disclosure", "Affirmation and Reassurance", "Providing Suggestions",
        "Information", "Others"
    ]
    for strategy in strategies:
        if strategy in baseline_metrics:
            print(f"{strategy:<40} {baseline_metrics[strategy]:>10.4f}")
    
    print("\n3. Ablation Study")
    print("-" * 80)
    print(f"{'Variant':<30} {'Macro F1':>12} {'Δ Macro':>12} {'Weighted F1':>12} {'Bias':>10}")
    print("-" * 80)
    print(f"{'Full Model (Baseline)':<30} {baseline_metrics['macro_f1']:>12.4f} {'-':>12} "
          f"{baseline_metrics['weighted_f1']:>12.4f} {baseline_metrics['preference_bias']:>10.4f}")
    
    for result in ablation_results:
        print(f"{result['variant']:<30} {result['macro_f1']:>12.4f} "
              f"{result['delta_macro_f1']:>+12.4f} {result['weighted_f1']:>12.4f} "
              f"{result['preference_bias']:>10.4f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EmoDynamiX-Telugu model")
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='xlmr-hg-telugu-esconv-telugu-checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--checkpoint_step', type=int, default=1500,
                        help='Checkpoint step to load')
    parser.add_argument('--hg_dim', type=int, default=256,
                        help='Heterogeneous graph dimension')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--telugu_erc_path', type=str,
                        default='telugu_erc_xlmroberta_trained_v2',
                        help='Path to Telugu ERC model')
    parser.add_argument('--parse_bs', type=int, default=1024,
                        help='SDDP batch size')
    parser.add_argument('--parse_ctx_len', type=int, default=32,
                        help='SDDP context length')
    parser.add_argument('--context_max_len', type=int, default=256,
                        help='Max context length for encoder')
    parser.add_argument('--erc_max_len', type=int, default=128,
                        help='Max length for ERC')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=114514,
                        help='Random seed')
    parser.add_argument('--skip_ablation', action='store_true',
                        help='Skip ablation study (faster evaluation)')
    
    args = parser.parse_args()
    
    # Setup
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("EmoDynamiX-Telugu Evaluation")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_dir}/checkpoint-{args.checkpoint_step}.pth")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Build model args
    from argparse import Namespace
    model_args = Namespace(
        seed=args.seed,
        hg_dim=args.hg_dim,
        erc_temperature=0.5,
        erc_mixed=1,
        telugu_erc_path=args.telugu_erc_path,
        parse_bs=args.parse_bs,
        parse_ctx_len=args.parse_ctx_len,
        context_max_len=args.context_max_len,
        erc_max_len=args.erc_max_len,
        exclude_others=0,
        feedback_threshold=0,
        dataset="esconv-telugu"
    )
    
    # Load model
    print("Loading model...")
    model = XLMRHeterogeneousGraphTelugu(model_args, lightmode=False)
    checkpoint_path = f"{args.checkpoint_dir}/checkpoint-{args.checkpoint_step}.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(f"Model loaded from {checkpoint_path}\n")
    
    # Load test data
    print("Loading test dataset...")
    test_set = ESConvTelugu("test", model_args)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn
    )
    id2label = test_set.id2label
    print(f"Test samples: {len(test_set)}\n")
    
    # 1. Main Evaluation
    print("="*80)
    print("MAIN EVALUATION")
    print("="*80)
    baseline_metrics, predictions, truths = evaluate_model(
        model, test_loader, device, id2label
    )
    
    # Save main metrics
    main_metrics_path = os.path.join(args.output_dir, 'main_metrics.json')
    with open(main_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMain metrics saved to {main_metrics_path}")
    
    # 2. Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = np.array(baseline_metrics['confusion_matrix'])
    label_names = [id2label[i] for i in range(len(id2label))]
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, label_names, cm_path, normalize=True)
    
    # 3. Ablation Study
    ablation_results = []
    if not args.skip_ablation:
        print("\n" + "="*80)
        print("ABLATION STUDY")
        print("="*80)
        ablation_results = run_ablation_study(
            model, test_loader, device, id2label, baseline_metrics
        )
        
        # Save ablation results
        ablation_path = os.path.join(args.output_dir, 'ablation_results.json')
        with open(ablation_path, 'w', encoding='utf-8') as f:
            json.dump(ablation_results, f, indent=2, ensure_ascii=False)
        print(f"\nAblation results saved to {ablation_path}")
    
    # 4. Print formatted table
    print_results_table(baseline_metrics, ablation_results)
    
    print(f"\nAll results saved to: {args.output_dir}/")
    print("  - main_metrics.json")
    print("  - confusion_matrix.png")
    if not args.skip_ablation:
        print("  - ablation_results.json")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
