# Backend Notes

This fixture is intentionally small so every source line and chunk can be checked by hand.

## epoll

Level-triggered epoll keeps reporting a ready descriptor until the application drains the condition. Edge-triggered epoll reports state changes, so nonblocking I/O must continue until EAGAIN.

```cpp
while (read(fd, buffer, sizeof(buffer)) > 0) {
    consume(buffer);
}
```

| Mode | Application rule |
| --- | --- |
| LT | Readiness may be reported again |
| ET | Drain nonblocking I/O until EAGAIN |

## Reactor

The Reactor waits for readiness events, dispatches the matching callback, and keeps blocking work away from the event loop. Worker threads may execute business tasks, but connection ownership and event registration remain coordinated by the loop thread.

Readiness is not the same as completion. An event says that an operation can make progress, while the callback still performs the nonblocking read or write and handles partial results. The loop must update its interest set when the connection changes from reading to writing or when buffered output has been drained.

Callbacks should stay short because one slow handler delays every other connection assigned to the same loop. CPU-heavy parsing or business work can be submitted to a worker pool, and the result is then handed back to the owning loop instead of letting worker threads mutate event registrations directly.

Connection lifetime also needs an explicit rule. A queued worker task may outlive the socket that created it, so callbacks should use a safe ownership mechanism and verify that the connection is still valid before publishing a result. Shutdown must remove the descriptor and cancel or ignore stale work without double-closing it.

Backpressure completes the design. When an output buffer grows beyond its configured high-water mark, the server should stop accepting unlimited work, pause reads where appropriate, or reject requests with an observable reason. This protects memory and keeps overload behavior explainable instead of allowing one slow client to exhaust the process.

![image-20260827203704467](C:\Users\1684541675\AppData\Roaming\Typora\typora-user-images\image-20260827203704467.png)