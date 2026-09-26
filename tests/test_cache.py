"""Cache and hashing tests."""

from manga_ai.utils.cache import ArtifactCache
from manga_ai.utils.hashing import cache_key, content_hash


def test_cache_skip():
    c = ArtifactCache(enabled=True)
    key = c.make_key("a", "b", "c", "0.1.0")
    assert c.should_skip("det", "a", "b", "c", "0.1.0", {"det": key}) is True
    assert c.should_skip("det", "a", "b", "c", "0.1.0", {}, force=False) is False
    assert c.should_skip("det", "a", "b", "c", "0.1.0", {"det": key}, force=True) is False


def test_cache_disabled():
    c = ArtifactCache(enabled=False)
    key = c.make_key("a", "b", "c", "0.1.0")
    assert c.should_skip("det", "a", "b", "c", "0.1.0", {"det": key}) is False


def test_content_hash_stable():
    assert content_hash({"x": 1}) == content_hash({"x": 1})
    assert content_hash([1, 2]) != content_hash([2, 1])
