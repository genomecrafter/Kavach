# Trained XLM-R router (demo model)

This folder holds the fine-tuned **XLM-R base** router selected by validation AUROC (`xlmr_s42`,
val AUROC 0.977, test AUROC 0.966). Metrics, predictions and training history are committed;
the weights are **not**, because GitHub rejects files over 100 MB:

| file | in git | size |
|---|---|---|
| `config.json`, `tokenizer*.json`, `router_config.json` | yes | small |
| `history.json`, `val_preds.jsonl`, `test_preds.jsonl` | yes | small |
| `model.safetensors` (fp16) | **no** | 556 MB |

To get the weights, do one of the following:

1. **Re-train:** run `dl/kaggle/kavach_router_kaggle.ipynb` on Kaggle (T4 x2, about 45 min) and copy
   `dl/best_router/` from its output zip into this folder.
2. **Download it** from the [GitHub Releases page](https://github.com/genomecrafter/Kavach/releases)
   (asset `model.safetensors`) and put it in this folder.

Without the weights, the demo automatically uses the TF-IDF router in `dl/runs/tfidf`.
