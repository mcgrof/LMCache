// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lmcache {
namespace connector {

// Key encoding constants — must match fs_l2_adapter.py
static constexpr const char* TMP_EXT = ".tmp";

// Bytes of reads kept outstanding against the device when num_read_workers is
// on and no explicit figure is configured.  Chosen as the value whose worst
// case across a local array, a single NVMe and 8 / 32 ms per-read latency
// is best; the sweep is in
// docs/design/v1/distributed/l2_adapters/fs_native_read_depth.md.
static constexpr size_t kDefaultReadMaxBytesInFlight = size_t{1536} << 20;
// A worker blocked on a full budget is woken once 1/kReadBudgetWakeSlices of
// it is free.
static constexpr size_t kReadBudgetWakeSlices = 8;
// Reads always admitted whatever the budget, so an object larger than half
// of it still overlaps with the next one instead of draining the device
// between them: one 1 GiB object alone reads ~40 GB/s on an array that
// two in flight bring to ~49.
static constexpr size_t kMinReadsInFlight = 2;

// Per-worker connection state for the FS connector.
// Each worker maintains its own I/O buffer for O_DIRECT.
struct WorkerFSConn {
  std::filesystem::path base_path;
  std::filesystem::path tmp_dir;  // empty if not configured
  bool use_odirect = false;
  size_t disk_block_size = 0;
  // If > 0, trigger filesystem readahead by issuing a small
  // initial read of this many bytes before reading the rest.
  size_t read_ahead_size = 0;
};

// Upper bound on HeldFdCache lock shards.
static constexpr size_t kHeldFdShards = 16;

// Bounded cache of read-only descriptors for published objects.
//
// An object file is published by an atomic rename and its key names its
// content, so a descriptor opened on a published file always reads a valid
// value for that key, even after another process evicts the file.  Reusing
// it skips the open and close (on NFS, an OPEN and a CLOSE round trip to
// the server) and lets exists() answer without a stat.  Entries are keyed
// by file name and O_DIRECT mode and reference counted: eviction and
// invalidation never close a descriptor a read still uses; it is closed
// when its last reader releases it.  All methods are thread-safe.
class HeldFdCache {
 public:
  explicit HeldFdCache(size_t capacity);
  ~HeldFdCache() { close_all(); }
  HeldFdCache(const HeldFdCache&) = delete;
  HeldFdCache& operator=(const HeldFdCache&) = delete;

  bool enabled() const { return capacity_ > 0; }

  // A descriptor for `path` with one reference, opened with `flags` when
  // not cached.  Throws if the open fails or the file holds fewer than
  // `min_size` bytes.
  int acquire(const std::string& name, const std::filesystem::path& path,
              int flags, size_t min_size);
  // Drops one reference.  `stale` removes the entry, so the next acquire
  // opens the file again.
  void release(const std::string& name, int flags, int fd, bool stale);
  // Whether a usable descriptor for `name` is cached.
  bool contains(const std::string& name);
  // Removes `name`'s entries, closing each once it has no readers.
  void invalidate(const std::string& name);
  // Closes every descriptor.  Only once no reader can be running.
  void close_all();
  std::map<std::string, uint64_t> stats();

 private:
  struct Entry {
    int fd = -1;
    size_t refs = 0;
    std::list<std::string>::iterator lru;
  };
  // Entries are split by file name across shards, each with its own lock
  // and LRU, so readers of different objects do not queue on one mutex.
  // The capacity is global: an insert that takes the cache over it evicts
  // from its own shard's LRU first, then from the others, so eviction
  // order is least recently used per shard.
  struct Shard {
    std::mutex mu;
    std::unordered_map<std::string, Entry> entries;
    std::list<std::string> lru;               // front: most recently used
    std::unordered_map<int, size_t> retired;  // fd -> remaining readers
    // Keys being opened.  Other readers of such a key wait on `opened`
    // for that open instead of sending their own.
    std::unordered_set<std::string> opening;
    std::condition_variable opened;
    uint64_t opens = 0, hits = 0, evictions = 0, invalidations = 0;
    uint64_t stale_drops = 0, deferred_closes = 0, open_waits = 0;
    void retire_locked(Entry& e);  // entry already removed from the map
    // Evicts unused entries, oldest first, while `total` exceeds `capacity`.
    void evict_locked(std::atomic<size_t>& total, size_t capacity);
  };
  static std::string key_of(const std::string& name, int flags);
  Shard& shard_of(const std::string& name);
  // Brings the cache back to its capacity from shards other than `skip`.
  void evict_others(const Shard* skip);
  bool release_locked(Shard& sh, const std::string& name, int flags, int fd,
                      bool stale);

  const size_t capacity_;
  std::vector<std::unique_ptr<Shard>> shards_;
  std::atomic<size_t> total_{0};  // cached entries, retired ones excluded
};

class FSConnector : public ConnectorBase<WorkerFSConn> {
 public:
  // num_read_workers: reader threads, and so the maximum reads in flight.
  //   Zero keeps the legacy path, where reads run on the worker threads
  //   and the depth against the device equals num_workers.
  // read_max_bytes_in_flight: bytes kept outstanding, connector-wide.
  //   Zero selects kDefaultReadMaxBytesInFlight when num_read_workers is
  //   positive.
  // held_fd_capacity: read-only descriptors kept open for reuse
  //   (HeldFdCache).  Zero, the default, opens and closes per read.
  // inline_read_keys: with a read pool and buffered reads, up to this many
  //   keys of each GET tile are read by the submitting worker itself when
  //   every byte is already in the page cache; the rest, and any key that
  //   is not cached, go to the pool.  Zero, the default, sends every key
  //   to the pool.
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0, int num_read_workers = 0,
              size_t read_max_bytes_in_flight = 0, size_t held_fd_capacity = 0,
              size_t inline_read_keys = 0);
  ~FSConnector() override;

  // The byte budget in force, or 0 when there is no read pool.
  size_t read_budget_bytes() const;

  // Held-descriptor counters: opens, hits, evictions, invalidations,
  // stale_drops, deferred_closes, open_waits, open_now and capacity.
  std::map<std::string, uint64_t> held_fd_stats();

  // Keys read inline from the page cache (inline_hits), keys tried
  // inline that went to the pool (inline_misses), and inline_unsupported,
  // 1 once the filesystem refused RWF_NOWAIT and inline reads stopped.
  std::map<std::string, uint64_t> read_stats();

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

  // With a read pool a GET batch stays one tile, as MooncakeConnector
  // does: the pool parallelises it, and splitting it across workers first
  // would cap reads in flight at num_workers.
  size_t choose_num_tiles(Op op, size_t num_items) const override;

  // Through the read pool when one is configured, otherwise the base
  // class's legacy path.  Per-key failures are reported the same way.
  void do_batch_get(WorkerFSConn& conn, const Request& req) override;

  // Stops the pool.  Only safe after the workers are joined: a worker may
  // be waiting on a tile only the readers can finish.
  void on_workers_stopped() override;

 private:
  // Completion state of one tile, on the submitting worker's stack.
  // Guarded by read_mu_, which also keeps it alive: the worker destroys it
  // only after taking that lock and seeing remaining == 0.
  struct ReadTile {
    std::vector<uint8_t> ok;  // one flag per object
    size_t remaining = 0;
    // Signalled once, by the reader that finishes the last object, so a
    // completion does not wake every worker waiting on some other tile.
    std::condition_variable done_cv;
  };

  // One object to read, owned by the submitting worker's stack and linked
  // into the read queue without allocating.
  struct ReadTask {
    const Request* req = nullptr;
    size_t index = 0;  // into req->keys and tile->ok
    ReadTile* tile = nullptr;
    ReadTask* next = nullptr;
  };

  // Read the tile's first inline_read_keys_ keys here when they are in the
  // page cache, hand the others to the readers one object at a time, each
  // dispatched as soon as the byte budget has room, and wait for all.
  void do_batch_get_pooled(WorkerFSConn& conn, const Request& req);

  // Queue the tile's keys `idx` for the readers and wake some.  Requires
  // read_mu_, held through `lk`, which a full byte budget releases while
  // it waits.
  void dispatch_locked(std::unique_lock<std::mutex>& lk, const Request& req,
                       ReadTile& tile, std::vector<ReadTask>& tasks,
                       const std::vector<size_t>& idx);

  // Reads an object only if all of it is in the page cache, without ever
  // waiting for the device.  False leaves the key to the pool.
  bool try_get_cached(const WorkerFSConn& conn, const std::string& key,
                      void* buf, size_t len);

  // Whether an object of `len` bytes may be dispatched now.  Requires
  // read_mu_.
  bool read_budget_admits(size_t len) const;

  // A reader's wakeup.  Owned by the connector, not the thread, so a
  // waker may notify it after releasing read_mu_ even if the reader has
  // since exited.
  struct ReaderSlot {
    std::condition_variable cv;
    bool woken = false;  // guarded by read_mu_
  };

  // Takes up to `n` idle readers, most recently idle first, and marks them
  // woken.  Requires read_mu_; the caller notifies them after releasing it.
  size_t take_idle_readers_locked(size_t n, ReaderSlot** out);

  void start_read_pool(int num_read_workers);
  void stop_read_pool();
  void read_thread_main(WorkerFSConn& conn, ReaderSlot& slot);

  // Reads `len` bytes of an object through a held descriptor, retrying
  // once with a fresh open if the cached one fails.
  void get_with_held_fd(const WorkerFSConn& conn, const std::string& filename,
                        const std::filesystem::path& file_path, int flags,
                        bool do_odirect, void* buf, size_t len);

  HeldFdCache held_fds_;
  std::string base_path_;
  std::string relative_tmp_dir_;
  bool use_odirect_;
  size_t disk_block_size_;
  size_t read_ahead_size_;
  size_t inline_read_keys_;
  // Set once preadv2(RWF_NOWAIT) is refused by the filesystem, so no
  // further inline attempts are made.
  std::atomic<bool> inline_unsupported_{false};
  std::atomic<uint64_t> inline_hits_{0};
  std::atomic<uint64_t> inline_misses_{0};

  // Read pool.  read_conns_ is sized once and never resized, so reader
  // references stay valid; read_max_bytes_in_flight_ is 0 without a pool
  // and never 0 with one.  Everything after read_mu_ is guarded by it.
  size_t read_max_bytes_in_flight_;
  std::vector<WorkerFSConn> read_conns_;
  std::vector<std::thread> read_threads_;
  std::vector<std::unique_ptr<ReaderSlot>> reader_slots_;
  std::mutex read_mu_;
  // Readers waiting for a task, most recently idle last.  Waking from the
  // back reuses the threads (and CPUs, and cache) that just ran, so a
  // large pool spreads light load over a few warm readers instead of
  // cycling through all of them.
  std::vector<ReaderSlot*> idle_readers_;
  // Budget room opened.  Signalled only once a slice of the budget is
  // free, so a worker blocked on a full budget wakes once per slice and
  // dispatches a run of objects, not once per finished read.
  std::condition_variable read_budget_cv_;
  size_t read_budget_waiters_ = 0;
  ReadTask* read_queue_head_ = nullptr;
  ReadTask* read_queue_tail_ = nullptr;
  size_t read_bytes_in_flight_ = 0;  // dispatched and not yet finished
  size_t reads_in_flight_ = 0;       // objects behind read_bytes_in_flight_
  bool read_stop_ = false;
};

}  // namespace connector
}  // namespace lmcache
