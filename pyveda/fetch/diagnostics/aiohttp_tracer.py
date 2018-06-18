import aiohttp
import collections

class BatchFetchTracer(object):
    def __init__(self, cache=None):
        if not cache:
            cache = collections.defaultdict(list)
        self._trace_cache = cache

    async def on_dns_resolvehost_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_dns_resolvehost_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self._trace_cache["dns_resolution_time"].append(elapsed)

    async def on_request_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_request_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self._trace_cache["request_times"].append(elapsed)

    async def on_request_exception(self, session, context, params):
        pass

    async def on_connection_queued_start(self, session, context, params):
        pass

    async def on_connection_queued_end(self, session, context, params):
        pass

    async def on_connection_create_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_connection_create_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self._trace_cache["connection_lifetimes"].append(elapsed)

    def _configure_tracer(self):
        trace_config = aiohttp.TraceConfig()
        trace_config.on_dns_resolvehost_start.append(self.on_dns_resolvehost_start)
        trace_config.on_dns_resolvehost_end.append(self.on_dns_resolvehost_end)
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_end.append(self.on_request_end)
        trace_config.on_request_exception.append(self.on_request_exception)
        trace_config.on_connection_queued_start.append(self.on_connection_queued_start)
        trace_config.on_connection_queued_end.append(self.on_connection_queued_end)
        trace_config.on_connection_create_start.append(self.on_connection_create_start)
        trace_config.on_connection_create_end.append(self.on_connection_create_end)
        return trace_config

