# `solvi.serve`

`solvi serve`: the questions over HTTP and MCP, a decider behind the System One API, and the MCP proxy that puts a
guard in front of another MCP server's `tools/call` (`run_proxy`, `Proxy`, `Upstream`; `solvi serve --guard
catalog.py:guard --upstream CMD`; stable since 1.1).

::: solvi.serve

## The proxy's old path, until 1.2

`solvi.experimental.mcp` re-exports the proxy's names, as the same objects, with an `ExperimentalWarning`.

::: solvi.experimental.mcp
    options:
      members: false
