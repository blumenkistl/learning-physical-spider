"""A Cadence brain in a physical spider: the learning spider's life in a MuJoCo world.

Scaffold: the body (``spider.model``, ``spider.body``) and the senses (``spider.senses``) are
here; the brain wrapper, the life, the nursery, the server and the page come over from
learning-spider as ``PORTING.md`` lays out. CPU only.
"""

from .body import Spider
from .model import DT, JOINTS, LEGS, MODEL_DIGEST

__all__ = ["DT", "JOINTS", "LEGS", "MODEL_DIGEST", "Spider"]
