import asyncio
import threading

from membench.systems.shared import run_async


def test_successive_calls_share_one_loop_so_loop_bound_locks_keep_working():
    lock = run_async(_make_lock())
    assert run_async(_use(lock)) == "used"


def test_each_thread_gets_its_own_loop():
    seen: dict[str, object] = {}

    def worker(name):
        seen[name] = run_async(_current_loop())

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["a"] is not seen["b"]


async def _make_lock():
    lock = asyncio.Lock()
    async with lock:
        pass
    return lock


async def _use(lock):
    async with lock:
        return "used"


async def _current_loop():
    return asyncio.get_running_loop()
