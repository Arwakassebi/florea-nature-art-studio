from florea.mem import ModelManager


def make(budget=5):
    mm = ModelManager(budget, verbose=False)
    for n, s in (("a", 1.0), ("b", 3.0), ("c", 2.6)):
        mm.register(n, lambda n=n: n, s)
    return mm


def test_lru_evicts_oldest():
    mm = make()
    mm.get("a"); mm.get("b"); mm.get("a"); mm.get("c")
    assert set(mm.loaded) == {"a", "c"}


def test_cache_hit_does_not_reload():
    mm = make()
    mm.get("a"); mm.get("a")
    assert mm.loads["a"] == 1


def test_free_fn_called_on_unload():
    calls = []
    mm = ModelManager(5, free_fn=lambda: calls.append(1), verbose=False)
    mm.register("a", lambda: object(), 1)
    mm.get("a"); mm.unload_all()
    assert calls == [1] and mm.status() == "aucun"
