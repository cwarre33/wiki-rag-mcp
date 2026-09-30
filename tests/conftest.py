import hashlib
import re

import numpy as np
import pytest

from wikirag.index import WikiIndex


class HashEmbedder:
    """Deterministic bag-of-words embedder so tests need no model download."""

    name = "test-hash"

    def encode(self, texts):
        out = np.zeros((len(texts), 256), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


PAGES = {
    "techniques/mbr-decoding.md": """---
title: MBR Decoding
visibility: public
related: [[wiki/models/byt5.md]], [[wiki/kaggle/deep-past.md]]
tags: [nlp, decoding]
---
# MBR Decoding

## How it works
Sample N candidates and return the candidate with the highest average similarity to all others, see [[wiki/models/byt5.md|ByT5]].
""",
    "tools/faiss.md": """---
title: FAISS
visibility: public
tags: [vector-search]
---
# FAISS

## Index types
FAISS IndexFlatIP performs exact inner product nearest neighbor search over embedding vectors.
""",
    "production-systems/secret.md": """---
title: Internal System
visibility: fls-internal
tags: [ops]
---
# Internal
Private operational details that must never be indexed or served.
""",
    "techniques/osint-thing.md": """---
title: OSINT Thing
visibility: public
tags: [osint, research]
---
# OSINT
Public but security-tagged, so it stays out of the index.
""",
    "open-questions/q.md": """---
title: Open Question
visibility: public
tags: [misc]
---
# Q
Public but in a redacted section, so it stays out of the index.
""",
}


@pytest.fixture
def wiki_dir(tmp_path):
    for rel, body in PAGES.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return tmp_path


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def index(wiki_dir, embedder):
    return WikiIndex.build(wiki_dir, embedder)
