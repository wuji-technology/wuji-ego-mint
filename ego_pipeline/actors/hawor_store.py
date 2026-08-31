
from __future__ import annotations

import ray


@ray.remote(max_concurrency=50)
class HaWorResultStore:
    
    def __init__(self):
        self._store:  dict = {}
        self._events: dict = {}

    async def put(self, scene: str, result: dict) -> None:
        self._store[scene] = result
        if scene in self._events:
            self._events[scene].set()

    async def wait_and_get(self, scene: str, timeout: float = 600) -> dict:
        import asyncio
        if scene not in self._store:
            ev = asyncio.Event()
            self._events[scene] = ev
            try:
                await asyncio.wait_for(ev.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self._events.pop(scene, None)
                raise TimeoutError(f'[pipeline]  {scene}; {timeout}.')
        self._events.pop(scene, None)
        return self._store.pop(scene)
