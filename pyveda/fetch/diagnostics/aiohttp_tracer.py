import aiohttp
import collections

class BatchFetchTracer(object):
    def __init__(self, cache=None):
        if not cache:
            cache = collections.defaultdict(list)
        self.cache = cache

    async def on_dns_resolvehost_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_dns_resolvehost_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self.cache["dns_resolution_time"].append(elapsed)

    async def on_request_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_request_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self.cache["request_times"].append(elapsed)

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
        self.cache["connection_lifetimes"].append(elapsed)


def batch_fetch_tracer(trace_config=None):
    if not trace_config:
        trace_config = aiohttp.TraceConfig()
    bft = BatchFetchTracer()
    trace_config.on_dns_resolvehost_start.append(bft.on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(bft.on_dns_resolvehost_end)
    trace_config.on_request_start.append(bft.on_request_start)
    trace_config.on_request_end.append(bft.on_request_end)
    trace_config.on_request_exception.append(bft.on_request_exception)
    trace_config.on_connection_queued_start.append(bft.on_connection_queued_start)
    trace_config.on_connection_queued_end.append(bft.on_connection_queued_end)
    trace_config.on_connection_create_start.append(bft.on_connection_create_start)
    trace_config.on_connection_create_end.append(bft.on_connection_create_end)

    return bft, trace_config

