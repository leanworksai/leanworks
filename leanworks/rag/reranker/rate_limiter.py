import time
import asyncio
import math
from typing import Optional


class _TokenBucket:
    def __init__(self, rate_per_sec: float, burst: float):
        self.rate = max(1e-6, rate_per_sec)         # tokens per second
        self.burst = max(burst, 1.0)
        self.tokens = self.burst                    # start full
        self.updated = time.monotonic()

    def _refill(self, now: float):
        dt = max(0.0, now - self.updated)
        self.updated = now
        self.tokens = min(self.burst, self.tokens + dt * self.rate)

    def time_until(self, amount: float, now: float) -> float:
        self._refill(now)
        if self.tokens >= amount:
            return 0.0
        deficit = amount - self.tokens
        return deficit / self.rate

    def take(self, amount: float, now: float):
        self._refill(now)
        if self.tokens < amount:
            raise RuntimeError("take() called without sufficient tokens")
        self.tokens -= amount


class DualRateLimiter:
    """
    Token-bucket limiter for Requests/minute (RPM) and Tokens/minute (TPM).
    - rpm: allowed requests per minute
    - tpm: allowed tokens per minute
    - bursts allow short spikes
    """
    def __init__(self, rpm: int, tpm: int,
                 burst_requests: Optional[int] = None,
                 burst_tokens: Optional[int] = None):
        self.req_bucket = _TokenBucket(rate_per_sec=rpm/60.0, burst=float(burst_requests or rpm))
        self.tok_bucket = _TokenBucket(rate_per_sec=tpm/60.0, burst=float(burst_tokens or tpm))
        self._lock = asyncio.Lock()

    async def acquire(self, est_tokens: int):
        # Reserve 1 request and est_tokens tokens atomically.
        while True:
            now = time.monotonic()
            async with self._lock:
                wait_r = self.req_bucket.time_until(1.0, now)
                wait_t = self.tok_bucket.time_until(float(est_tokens), now)
                wait = max(wait_r, wait_t)
                if wait <= 0.0:
                    self.req_bucket.take(1.0, now)
                    self.tok_bucket.take(float(est_tokens), now)
                    return
            await asyncio.sleep(min(wait, 2.0))  # sleep a bit, then re-check

    def observe_headers(self, headers: dict):
        """
        Optional: call this with HTTP headers after each response or RateLimit error.
        If the SDK exposes 'Retry-After' or x-ratelimit-reset-* headers, you could
        fast-forward the buckets' clocks or conservatively sleep in the caller.
        Provided as a hook; safe to no-op if unavailable.
        """
        return


def _approx_token_count(text: str) -> int:
    # ~4 chars/token heuristic; replace with `tiktoken` if available
    return max(1, math.ceil(len(text) / 4))


def _estimate_prompt_tokens(query: str, docs: list[str], overhead_tokens: int = 80) -> int:
    return overhead_tokens + _approx_token_count(query) + sum(_approx_token_count(d) for d in docs)


def _adaptive_batch_size(docs_remaining: int,
                         avg_doc_tokens: int,
                         limiter: DualRateLimiter,
                         hard_cap: int = 8,
                         overhead_tokens: int = 80,
                         target_latency_s: float = 1.2) -> int:
    """
    Choose a batch size that fits into the near-term token allowance.
    We look at how many tokens the token-bucket can refill during the next ~latency window.
    """
    # How many tokens will be (re)available soon?
    # Approximate near-term budget = current tokens + next 'target_latency_s' worth of refill.
    tok_refill_rate = limiter.tok_bucket.rate                 # tokens/sec
    available_soon = limiter.tok_bucket.tokens + tok_refill_rate * target_latency_s
    # Reserve some headroom
    available_soon = max(0, available_soon * 0.9 - overhead_tokens)

    if available_soon <= 0:
        return 1

    max_by_tokens = int(max(1, available_soon // max(1, avg_doc_tokens)))
    return max(1, min(hard_cap, docs_remaining, max_by_tokens))


def _jitter_backoff(base: float, cap: float, attempt: int) -> float:
    # "decorrelated jitter" (AWS/ExponentialBackoffAndJitter)
    import random
    if attempt <= 0: 
        return base
    sleep = min(cap, random.uniform(base, base * (2 ** attempt)))
    return sleep
