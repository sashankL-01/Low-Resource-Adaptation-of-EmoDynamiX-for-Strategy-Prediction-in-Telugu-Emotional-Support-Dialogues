# Telugu Model Optimization Notes

## Precompute ERC Logits
Run the frozen Telugu ERC model over all seeker concatenated texts once and store a tensor per dialogue sample:
- Shape: (num_seeker_turns_in_sample, 7)
- Serialize with `torch.save` or embed into preprocessed pickle similar to `esconv_preprocessed`.
- At runtime provide `erc_logits` (will be temperature + softmaxed inside model) or directly provide graph-space `erc_embeddings` (shape: total_seeker_turns, hg_dim) to skip prototype mixing.

## Precompute Parsed Dialogues
If English discourse parser speed is a bottleneck:
- For each sample store `parsed_dialogue`: list of `(head, tail, relation_id)` triples.
- Switch model init to `lightmode=True` and pass `parsed_dialogue` in batches to avoid parser calls.

## Batch Collation for Precomputed Embeddings
Extend Telugu dataset collate to include:
```python
'parsed_dialogue': [...],          # list per sample
'erc_logits': torch.cat([...], 0)  # seeker logits concatenated across batch
```
All indices logic in model relies on concatenation order; keep sample order stable.

## Context Embedding Caching
XLM-R large is heavy. Cache `context_embeddings` keyed by a hash of `dialogue_history`.
- Use LRU dictionary: `cache_size ~ 50k` depending on RAM.
- If cached, skip encoder forward; directly stack with graph embeddings.

## Mixed vs Hard Emotion Setting
- Mixed (`erc_mixed=1`) uses soft combination of prototypes; smoother gradients in graph layers.
- Hard tags (`erc_mixed=0`) faster (embedding lookup) but may reduce expressiveness.

## Gradient/Memory Tweaks
- Use `torch.cuda.amp.autocast()` + GradScaler for half precision (requires adding around training loop) to cut memory ~40%.
- If OOM, reduce `hg_dim` (e.g., 384) and keep prototypes size aligned.

## Edge Pruning (Optional)
- Remove low-attention edges after first epoch and rebuild a sparse edge list for later epochs to accelerate RGATConv.

## Checkpoint Strategy
- Save only model state dict; XLM-R + ERC are frozen => optionally exclude their weights to shrink size by filtering keys whose prefix matches encoder or erc_model modules.

## Minimal Preprocessing Script Skeleton
```python
# pseudo-code
for sample in telugu_dataset_samples:
    seeker_concat = build_seeker_text(sample)
    logits = erc_model(tokenizer(seeker_concat, return_tensors='pt', ...))['logits']
    sample['erc_logits'] = logits.cpu()
    sample['parsed_dialogue'] = parser.parse(sample_dialogue_turn_dicts)
# dump pickle with list of enhanced samples
```

