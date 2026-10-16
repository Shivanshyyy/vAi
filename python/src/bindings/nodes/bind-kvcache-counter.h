#ifndef PYVAI_BIND_KVCACHE_COUNTER_H
#define PYVAI_BIND_KVCACHE_COUNTER_H

// KVCacheCounter — Python-only wrapper around the uint32 KV-cache token-counter
// buffer the C++ models manage as a raw eva::Buffer. Not a node; it lives in
// this shared header because the KV-aware nodes take it via
// set_kv_cache_counter(), and is registered once, before them.
#include "bind-nodes.h"

#include "vai/nodes/node-common.h"   // eva::Buffer, vai::getVaiDevice()

namespace pyvai {

struct KVCacheCounter
{
    eva::Buffer buf;

    KVCacheCounter() = default;
    explicit KVCacheCounter(eva::Buffer b) : buf(b) {}

    // eva handles need an explicit destroy() (idempotent); nodes pin the
    // counter via keep_alive, so this never runs under a live reference.
    ~KVCacheCounter() { buf.destroy(); }
    KVCacheCounter(const KVCacheCounter&) = delete;
    KVCacheCounter& operator=(const KVCacheCounter&) = delete;
};

void bindKVCacheCounter(nb::module_& m);

}   // namespace pyvai

#endif  // PYVAI_BIND_KVCACHE_COUNTER_H
