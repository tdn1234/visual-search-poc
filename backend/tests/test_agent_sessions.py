"""SessionStore: Redis-backed history with a message cap and TTL (fake Redis)."""

from app.services.agent_sessions import SessionStore


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttls[key] = ex


def test_unknown_session_is_empty():
    assert SessionStore(FakeRedis()).load("nope") == []


def test_round_trip_sets_ttl():
    redis = FakeRedis()
    store = SessionStore(redis, ttl_seconds=60, max_messages=10)

    store.append_turn("s1", [], "hi", "hello")

    assert store.load("s1") == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert redis.ttls["agent:session:s1"] == 60


def test_history_is_capped_to_the_most_recent_messages():
    store = SessionStore(FakeRedis(), max_messages=4)
    history = []
    for i in range(5):
        store.append_turn("s", history, f"q{i}", f"a{i}")
        history = store.load("s")

    assert [m["content"] for m in history] == ["q3", "a3", "q4", "a4"]


def test_corrupt_or_foreign_entries_are_ignored():
    redis = FakeRedis()
    redis.data["agent:session:s"] = "not json"
    assert SessionStore(redis).load("s") == []

    redis.data["agent:session:s"] = '[{"role": "system", "content": "x"}, {"role": "user", "content": "ok"}]'
    assert SessionStore(redis).load("s") == [{"role": "user", "content": "ok"}]
