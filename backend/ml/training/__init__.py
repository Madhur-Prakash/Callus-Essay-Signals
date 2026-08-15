"""Training pipeline stages.

    prepare_dataset   -> data/processed/corpus.jsonl + splits.json + manifest.json
    extract_features  -> data/processed/features.npz
    train             -> ml/artifacts/*.joblib + model_metadata.json
"""
