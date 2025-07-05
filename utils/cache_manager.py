class CacheManager:
    def __init__(self, max_size_mb=100, duration_minutes=15):
        self.cache = {}
        self.max_size_mb = max_size_mb
        self.duration_minutes = duration_minutes

    def set(self, key, value):
        if self._is_cache_full():
            self._evict_cache()
        self.cache[key] = value

    def get(self, key):
        return self.cache.get(key)

    def _is_cache_full(self):
        # Implement logic to check if cache size exceeds max_size_mb
        return False

    def _evict_cache(self):
        # Implement logic to evict old cache entries
        pass

    def clear(self):
        self.cache.clear()