"""Read newer list-style special-token metadata with Transformers 4.x too.

Create a private tokenizer view, never edit the checkpoint or invent token IDs.
Both our HF loader and vLLM must read the same view: fixing just the HF kwargs
would leave vLLM's independent tokenizer load broken.
"""

from __future__ import annotations

import json
from pathlib import Path


def _content(token):
    if isinstance(token, str):
        return token
    if isinstance(token, dict) and isinstance(token.get("content"), str):
        return token["content"]
    raise ValueError(f"Invalid serialized special token: {token!r}")


def _saved_vocabulary(source):
    backend = source / "tokenizer.json"
    if backend.is_file():
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(backend)).get_vocab()
    vocab = source / "vocab.json"
    if not vocab.is_file():
        raise ValueError(
            f"Cannot verify special-token IDs in {source}: expected tokenizer.json or vocab.json. "
            "Refusing to add/remap tokens while converting metadata."
        )
    result = json.loads(vocab.read_text(encoding="utf-8"))
    added = source / "added_tokens.json"
    if added.is_file():
        result.update(json.loads(added.read_text(encoding="utf-8")))
    config = source / "tokenizer_config.json"
    if config.is_file():
        for index, token in (
            json.loads(config.read_text(encoding="utf-8"))
            .get("added_tokens_decoder", {})
            .items()
        ):
            result[_content(token)] = int(index)
    return result


def prepare_tokenizer_view(source, destination):
    """Return original path unless list-style extra_special_tokens needs a view.

    V4 uses extra_special_tokens for a *named mapping*; V5 also serializes an
    anonymous list there. The shared backward-compatible representation is
    additional_special_tokens, with the added-token decoder/backend unchanged.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    patched, extras = {}, []
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = source / name
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        tokens = metadata.get("extra_special_tokens")
        if not isinstance(tokens, list):
            continue
        metadata.pop("extra_special_tokens")
        additional = metadata.get("additional_special_tokens", [])
        if not isinstance(additional, list):
            raise ValueError(f"{path}: additional_special_tokens must be a list")
        merged = list(additional)
        known = {_content(token) for token in merged}
        for token in tokens:
            content = _content(token)
            extras.append(content)
            if content not in known:
                merged.append(token)
                known.add(content)
        if merged or "additional_special_tokens" in metadata:
            metadata["additional_special_tokens"] = merged
        patched[name] = metadata
    if not patched:
        return source
    if extras:
        vocabulary = _saved_vocabulary(source)
        missing = sorted(set(extras) - vocabulary.keys())
        if missing:
            raise ValueError(
                f"{source}: extra_special_tokens are missing from the saved vocabulary: {missing}. "
                "Refusing to invent/remap checkpoint token IDs."
            )
    destination.mkdir(parents=True, exist_ok=False)
    for asset in source.iterdir():
        if asset.name in patched:
            (destination / asset.name).write_text(
                json.dumps(patched[asset.name], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            # No model-weight copy; preserve custom code, backend, templates and
            # any model-specific tokenizer assets without guessing filenames.
            (destination / asset.name).symlink_to(
                asset, target_is_directory=asset.is_dir()
            )
    (destination / "bellmanopd_tokenizer_compat.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "patched_files": list(patched),
                "conversion": "list extra_special_tokens -> additional_special_tokens",
                "checkpoint_modified": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def verify_tokenizer_view(tokenizer, source):
    """Fail before inference if loading the view changed the vocabulary/IDs."""
    expected = _saved_vocabulary(Path(source))
    if tokenizer.get_vocab() != expected or len(tokenizer) != len(expected):
        raise ValueError(
            f"Tokenizer metadata compatibility changed vocabulary/IDs relative to {source}; "
            "refusing to evaluate a different tokenizer."
        )
