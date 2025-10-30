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
    Extract word-level embeddings and align with original tokens.

    Returns a list of dicts for each sentence:
        [{'token': 'I', 'embedding': ..., 'label': 0}, ...]

    This makes it easy to trace which words were kept or dropped.
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

    result_sentences = []  # Each element is a list of {'token', 'embedding', 'label'}
    if all_labels is None:
        all_labels = [None] * len(all_tokens)

    with torch.no_grad():
        for i, (tokens, labels) in enumerate(zip(all_tokens, all_labels)):
            print(f"\nProcessing sentence {i+1}/{len(all_tokens)}: {' '.join(tokens)}")

            # Tokenize
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=False,
            )

            # Move to GPU
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_size)

            # Get word_ids from the encoded object (✅ correct way)
            word_ids = encoded.word_ids(batch_index=0)  # List[int or None], length = subword sequence

            # Extract one embedding per original word
            sentence_data = []
            previous_word_idx = None

            for token_position, word_idx in enumerate(word_ids):
                if word_idx is None:
                    print(f"  Ignored special token at position {token_position} (e.g., [CLS], [SEP])")
                    continue

                if word_idx != previous_word_idx:
                    # First subword of this word
                    emb = last_hidden_state[0, token_position, :].cpu()  # Move to CPU
                    token_str = tokens[word_idx]
                    label = labels[word_idx] if labels else None

                    sentence_data.append({
                        "token": token_str,
                        "embedding": emb,
                        "label": label,
                        "subword_position": token_position,
                    })
                    print(f"  Kept: '{token_str}' -> embedding from subword #{token_position}")
                else:
                    # Subsequent subword (e.g., ##is), skipped
                    subword_token = tokenizer.convert_ids_to_tokens(input_ids[0][token_position].item())
                    print(f"  Skipped subword: '{subword_token}' (part of '{tokens[word_idx]}')")

                previous_word_idx = word_idx

            result_sentences.append(sentence_data)

    print(f"\n✅ Extraction completed. Processed {len(result_sentences)} sentences.")
    return result_sentences
