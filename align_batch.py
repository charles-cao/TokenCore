from transformers import AutoTokenizer, AutoModel
import torch

def extract_word_embeddings(
    model,
    tokenizer,
    all_tokens,
    all_labels=None,
    model_name="bert-base-cased",
    device=None,
    max_length=128,
):
    """
    Extract word-level embeddings with correct alignment using batch inference.
    Fixes the 'word_ids' issue by storing alignment info before batching.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    local_cache_dir = "D:\model"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir)
    if model is None:
        model = AutoModel.from_pretrained(model_name, cache_dir=local_cache_dir)
    model = model.to(device).eval()

    # === Step 1: Tokenize each sentence and save word_ids ===
    input_ids_list = []
    attention_mask_list = []
    word_ids_list = []  # ✅ Save word_ids for each sentence BEFORE batching

    for tokens in all_tokens:
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            return_tensors=None,  # Return dict of lists, not tensor
        )
        input_ids_list.append(torch.tensor(encoded["input_ids"]))
        attention_mask_list.append(torch.tensor(encoded["attention_mask"]))
        # ✅ Save word_ids for this sentence
        word_ids_list.append(tokenizer.word_ids(batch_index=0))  # Only works on single sample

    # === Step 2: Pad and create batch ===
    from torch.nn.utils.rnn import pad_sequence

    input_ids_batch = pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)

    attention_mask_batch = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    ).to(device)

    # === Step 3: Forward pass (batched) ===
    with torch.no_grad():
        outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

    # === Step 4: Extract word-level embeddings for each sentence ===
    embeddings_list = []
    extracted_tokens = []

    for i in range(len(all_tokens)):
        tokens = all_tokens[i]
        word_ids = word_ids_list[i]  # ✅ Use pre-saved word_ids
        prev_word_idx = None
        word_embs = []

        for j, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:
                emb = last_hidden_state[i, j, :].cpu()  # Take embedding from batch output
                word_embs.append(emb)
            prev_word_idx = word_idx

        word_embs = torch.stack(word_embs)  # (num_words, hidden_size)
        embeddings_list.append(word_embs)
        extracted_tokens.append(tokens)

    print(f"✅ Finished extracting embeddings for {len(embeddings_list)} sentences.")

    return {
        "embeddings": embeddings_list,
        "tokens": extracted_tokens,
        "labels": all_labels,
    }

