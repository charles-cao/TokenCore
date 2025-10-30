from transformers import AutoTokenizer, AutoModel
import torch

def extract_word_embeddings(
    model,
    tokenizer,
    all_tokens,
    all_labels=None,
    model_name="bert-base-cased",
    device=None,
):
    """
    Extract word-level embeddings from sentences using a transformer model.
    Aligns each embedding with the original token and label.

    Args:
        model: Pretrained model (e.g., BERT). If None, loaded automatically.
        tokenizer: Tokenizer. If None, loaded automatically.
        all_tokens: List of tokenized sentences, e.g., [['I', 'am'], ['Good', 'morning']]
        all_labels: Optional list of labels for each token (0=normal, 1=anomaly)
        model_name: Name of the pretrained model
        device: Device to use ('cuda' or 'cpu'). Auto-detect if None.

    Returns:
        embeddings_list: List of tensors, each shape (num_words, hidden_size)
        all_labels: The input labels (returned for alignment)
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer if not provided
    local_cache_dir = "D:\model"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir)
    if model is None:
        model = AutoModel.from_pretrained(model_name, cache_dir=local_cache_dir)
    model = model.to(device).eval()

    embeddings_list = []
    if all_labels is None:
        all_labels = [None] * len(all_tokens)

    with torch.no_grad():
        for i, (tokens, labels) in enumerate(zip(all_tokens, all_labels)):
            print(f"Processing sentence {i+1}/{len(all_tokens)}: {' '.join(tokens)}")

            # Tokenize
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=False,
            )

            # Move input tensors to GPU
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_size)

            # Get word ids for alignment
            word_ids = encoded.word_ids(batch_index=0)

            # Extract one embedding per original word (first subword)
            word_embeddings = []
            prev_word_idx = None

            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue  # Ignore special tokens like [CLS], [SEP]
                if word_idx != prev_word_idx:
                    # This is the first subword of a new word
                    emb = last_hidden_state[0, token_idx, :]  # (hidden_size,)
                    emb = emb.cpu()  # Move back to CPU to save GPU memory
                    word_embeddings.append(emb)
                prev_word_idx = word_idx

            # Stack into a single tensor: (num_words, hidden_size)
            word_embeddings = torch.stack(word_embeddings)
            embeddings_list.append(word_embeddings)

            print(f"  -> Got {word_embeddings.shape[0]} word embeddings")

    print(f"\n✅ Finished. Total sentences processed: {len(embeddings_list)}")
    return embeddings_list, all_labels






