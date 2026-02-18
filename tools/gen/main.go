package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ─── Content pools ────────────────────────────────────────────────────────────

var topics = []string{
	"Go Performance", "Distributed Systems", "Database Internals", "Frontend Architecture",
	"Compiler Design", "Operating Systems", "Network Protocols", "Security Engineering",
	"Machine Learning Systems", "Data Structures", "Algorithm Design", "Cloud Infrastructure",
	"Observability Engineering", "API Design", "Concurrency Patterns", "Memory Management",
	"Functional Programming", "Type Systems", "Testing Strategies", "DevOps Practices",
	"Kubernetes Internals", "WebAssembly", "Rust Ownership Model", "Applied Cryptography",
	"Event-Driven Architecture", "CQRS & Event Sourcing", "Service Mesh", "eBPF",
	"Zero-Copy I/O", "Lock-Free Data Structures", "Cache Coherence", "SIMD Optimization",
	"Profiling & Tracing", "Static Analysis", "Formal Verification", "Consensus Algorithms",
	"Bloom Filters & Probabilistic Structures", "LSM Trees", "B-Trees & Variants", "Skip Lists",
}

var tagPool = []string{
	"Go", "Rust", "Performance", "Systems", "Databases", "Frontend", "Backend",
	"Concurrency", "Algorithms", "Security", "Cloud", "DevOps", "Testing",
	"Architecture", "Networking", "Compilers", "Memory", "Observability",
	"Kubernetes", "WebAssembly", "Cryptography", "Distributed", "API",
}

var quotes = []struct{ text, author, source string }{
	{"Premature optimization is the root of all evil.", "Donald Knuth", "The Art of Computer Programming"},
	{"Make it work, make it right, make it fast.", "Kent Beck", "Extreme Programming"},
	{"Simplicity is the ultimate sophistication.", "Leonardo da Vinci", "Notebooks"},
	{"Programs must be written for people to read, and only incidentally for machines to execute.", "Harold Abelson", "SICP"},
	{"The best code is no code at all.", "Jeff Atwood", "Coding Horror"},
	{"Stability comes from designing for failures, not avoiding them.", "Michael T. Nygard", "Release It!"},
	{"A distributed system is one in which the failure of a computer you didn't even know existed can render your own computer unusable.", "Leslie Lamport", "ACM Queue"},
	{"Correctness is clearly the prime quality. If a system does not do what it is supposed to do, then everything else about it matters little.", "Tony Hoare", "Hints on Programming Language Design"},
	{"The purpose of abstracting is not to be vague, but to create a new semantic level in which one can be absolutely precise.", "Edsger Dijkstra", "Selected Writings on Computing"},
	{"Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it.", "Brian Kernighan", "The Elements of Programming Style"},
	{"Any fool can write code that a computer can understand. Good programmers write code that humans can understand.", "Martin Fowler", "Refactoring"},
	{"There are only two hard things in Computer Science: cache invalidation and naming things.", "Phil Karlton", "Netscape"},
	{"Measuring programming progress by lines of code is like measuring aircraft building progress by weight.", "Bill Gates", "Microsoft Memo"},
	{"The most dangerous kind of waste is the waste we do not recognize.", "Shigeo Shingo", "A Study of the Toyota Production System"},
	{"In theory, theory and practice are the same. In practice, they are not.", "Yogi Berra", "Various"},
	{"The art of programming is the art of organizing complexity.", "Edsger Dijkstra", "Notes on Structured Programming"},
	{"First, solve the problem. Then, write the code.", "John Johnson", "Various"},
	{"Code is like humor. When you have to explain it, it's bad.", "Cory House", "Various"},
	{"Fix the cause, not the symptom.", "Steve Maguire", "Writing Solid Code"},
	{"Walking on water and developing software from a specification are easy if both are frozen.", "Edward V. Berard", "Life-Cycle Approaches"},
}

var codeSnippets = []struct{ lang, code string }{
	{"go", `// WorkerPool processes jobs concurrently with bounded parallelism.
type WorkerPool struct {
	jobs    chan Job
	results chan Result
	wg      sync.WaitGroup
}

func NewWorkerPool(size int) *WorkerPool {
	p := &WorkerPool{
		jobs:    make(chan Job, size*2),
		results: make(chan Result, size*2),
	}
	for i := 0; i < size; i++ {
		p.wg.Add(1)
		go p.worker()
	}
	return p
}

func (p *WorkerPool) worker() {
	defer p.wg.Done()
	for job := range p.jobs {
		p.results <- job.Process()
	}
}

func (p *WorkerPool) Submit(j Job) { p.jobs <- j }
func (p *WorkerPool) Close()       { close(p.jobs); p.wg.Wait(); close(p.results) }`},

	{"go", `// sync.Pool eliminates GC pressure on hot-path allocations.
var bufPool = sync.Pool{
	New: func() any { return bytes.NewBuffer(make([]byte, 0, 4096)) },
}

func processRequest(data []byte) ([]byte, error) {
	buf := bufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer bufPool.Put(buf)

	if err := json.NewEncoder(buf).Encode(data); err != nil {
		return nil, err
	}
	// Return a copy — the pool buffer is reused.
	out := make([]byte, buf.Len())
	copy(out, buf.Bytes())
	return out, nil
}`},

	{"go", `// Lock-free stack using atomic compare-and-swap.
type node[T any] struct {
	val  T
	next unsafe.Pointer
}

type Stack[T any] struct{ head unsafe.Pointer }

func (s *Stack[T]) Push(v T) {
	n := &node[T]{val: v}
	for {
		old := atomic.LoadPointer(&s.head)
		n.next = old
		if atomic.CompareAndSwapPointer(&s.head, old, unsafe.Pointer(n)) {
			return
		}
	}
}

func (s *Stack[T]) Pop() (T, bool) {
	for {
		old := atomic.LoadPointer(&s.head)
		if old == nil {
			var zero T
			return zero, false
		}
		n := (*node[T])(old)
		if atomic.CompareAndSwapPointer(&s.head, old, n.next) {
			return n.val, true
		}
	}
}`},

	{"go", `// Sieve of Eratosthenes — cache-friendly bit-packed variant.
func sieve(limit int) []int {
	words := (limit + 63) / 64
	bits := make([]uint64, words)

	for i := 2; i*i <= limit; i++ {
		if bits[i/64]>>(i%64)&1 == 0 {
			for j := i * i; j <= limit; j += i {
				bits[j/64] |= 1 << (j % 64)
			}
		}
	}
	primes := make([]int, 0, limit/int(math.Log(float64(limit))+1))
	for i := 2; i <= limit; i++ {
		if bits[i/64]>>(i%64)&1 == 0 {
			primes = append(primes, i)
		}
	}
	return primes
}`},

	{"go", `// Generic ring buffer — zero allocation after construction.
type RingBuffer[T any] struct {
	buf        []T
	head, tail int
	size, cap  int
}

func NewRingBuffer[T any](capacity int) *RingBuffer[T] {
	return &RingBuffer[T]{buf: make([]T, capacity), cap: capacity}
}

func (r *RingBuffer[T]) Push(v T) bool {
	if r.size == r.cap { return false }
	r.buf[r.tail] = v
	r.tail = (r.tail + 1) % r.cap
	r.size++
	return true
}

func (r *RingBuffer[T]) Pop() (T, bool) {
	if r.size == 0 { var z T; return z, false }
	v := r.buf[r.head]
	r.head = (r.head + 1) % r.cap
	r.size--
	return v, true
}`},

	{"go", `// TTL cache with O(1) eviction using a doubly-linked list.
type entry struct {
	key     string
	value   any
	expiry  time.Time
	prev, next *entry
}

type TTLCache struct {
	mu      sync.Mutex
	items   map[string]*entry
	head    *entry // most-recently used
	tail    *entry // least-recently used
	maxSize int
}

func (c *TTLCache) Get(key string) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.items[key]
	if !ok || time.Now().After(e.expiry) {
		if ok { c.evict(e) }
		return nil, false
	}
	c.moveToFront(e)
	return e.value, true
}`},

	{"rust", `// Ownership-safe concurrent queue using Arc + Mutex.
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Condvar};

pub struct Queue<T> {
    inner: Arc<(Mutex<VecDeque<T>>, Condvar)>,
}

impl<T: Send + 'static> Queue<T> {
    pub fn new() -> Self {
        Queue { inner: Arc::new((Mutex::new(VecDeque::new()), Condvar::new())) }
    }

    pub fn push(&self, item: T) {
        let (lock, cvar) = &*self.inner;
        lock.lock().unwrap().push_back(item);
        cvar.notify_one();
    }

    pub fn pop(&self) -> T {
        let (lock, cvar) = &*self.inner;
        let mut q = cvar.wait_while(lock.lock().unwrap(), |q| q.is_empty()).unwrap();
        q.pop_front().unwrap()
    }
}`},

	{"rust", `// Zero-cost abstraction: compile-time state machine.
struct Locked;
struct Unlocked;

struct Safe<State> {
    data: Vec<u8>,
    _state: std::marker::PhantomData<State>,
}

impl Safe<Locked> {
    fn new(data: Vec<u8>) -> Self {
        Safe { data, _state: std::marker::PhantomData }
    }
    fn unlock(self, _key: &str) -> Safe<Unlocked> {
        Safe { data: self.data, _state: std::marker::PhantomData }
    }
}

impl Safe<Unlocked> {
    fn read(&self) -> &[u8] { &self.data }
    fn lock(self) -> Safe<Locked> {
        Safe { data: self.data, _state: std::marker::PhantomData }
    }
}`},

	{"sql", `-- Efficient pagination using keyset (seek method) instead of OFFSET.
-- OFFSET scans and discards rows; keyset jumps directly to the cursor.
SELECT
    p.id,
    p.title,
    p.published_at,
    u.username  AS author,
    COUNT(c.id) AS comments
FROM posts p
JOIN users    u ON u.id = p.author_id
LEFT JOIN comments c ON c.post_id = p.id
WHERE p.published_at < :cursor          -- keyset cursor
  AND p.status = 'published'
GROUP BY p.id, p.title, p.published_at, u.username
ORDER BY p.published_at DESC, p.id DESC
LIMIT :page_size;

-- Index to support this query:
CREATE INDEX idx_posts_cursor
    ON posts (published_at DESC, id DESC)
    WHERE status = 'published';`},

	{"sql", `-- Recursive CTE: compute the transitive closure of a graph.
WITH RECURSIVE reachable(from_node, to_node, depth) AS (
    -- Base case: direct edges
    SELECT from_node, to_node, 1
    FROM   edges

    UNION ALL

    -- Recursive step: extend paths
    SELECT r.from_node, e.to_node, r.depth + 1
    FROM   reachable r
    JOIN   edges e ON e.from_node = r.to_node
    WHERE  r.depth < 10   -- cycle guard
)
SELECT DISTINCT from_node, to_node, MIN(depth) AS shortest_path
FROM   reachable
GROUP  BY from_node, to_node
ORDER  BY from_node, shortest_path;`},

	{"yaml", `# Production-grade Kubernetes Deployment with resource budgets,
# liveness/readiness probes, and pod disruption budget.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  labels: { app: api-server, tier: backend }
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
  selector:
    matchLabels: { app: api-server }
  template:
    metadata:
      labels: { app: api-server }
    spec:
      containers:
      - name: api
        image: registry.example.com/api:v2.4.1
        ports: [{ containerPort: 8080 }]
        resources:
          requests: { cpu: 250m, memory: 256Mi }
          limits:   { cpu: 1000m, memory: 512Mi }
        livenessProbe:
          httpGet: { path: /healthz, port: 8080 }
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet: { path: /ready, port: 8080 }
          initialDelaySeconds: 5
          periodSeconds: 5`},

	{"bash", `#!/usr/bin/env bash
# Robust deployment script with rollback on failure.
set -euo pipefail
trap 'echo "FAILED at line $LINENO" >&2; rollback' ERR

DEPLOY_TAG="${1:?Usage: deploy.sh <image-tag>}"
NAMESPACE="production"
DEPLOYMENT="api-server"

rollback() {
    echo "Rolling back to previous revision..."
    kubectl rollout undo "deployment/$DEPLOYMENT" -n "$NAMESPACE"
    exit 1
}

echo "Deploying $DEPLOY_TAG to $NAMESPACE..."
kubectl set image "deployment/$DEPLOYMENT" \
    "api=registry.example.com/api:$DEPLOY_TAG" \
    -n "$NAMESPACE"

kubectl rollout status "deployment/$DEPLOYMENT" \
    -n "$NAMESPACE" --timeout=5m

echo "Deployment successful."`},

	{"go", `// HTTP middleware chain — composable, allocation-free.
type Middleware func(http.Handler) http.Handler

func Chain(h http.Handler, middlewares ...Middleware) http.Handler {
	// Apply in reverse so the first middleware is outermost.
	for i := len(middlewares) - 1; i >= 0; i-- {
		h = middlewares[i](h)
	}
	return h
}

func Logger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(rw, r)
		log.Printf("%s %s %d %v", r.Method, r.URL.Path, rw.status, time.Since(start))
	})
}

func RateLimit(rps int) Middleware {
	limiter := rate.NewLimiter(rate.Limit(rps), rps)
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !limiter.Allow() {
				http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}`},
}

var paragraphs = []string{
	"When building high-throughput systems, the difference between a naive implementation and an optimized one can be several orders of magnitude. Understanding where your bottlenecks lie is the first step toward meaningful improvement. Without measurement, optimization is guesswork — and guesswork at scale is expensive.",
	"The key insight here is that most performance problems are not where developers expect them to be. Profiling before optimizing is not just good advice — it is the only rational approach to performance engineering. The CPU flame graph rarely lies, and it almost always surprises.",
	"Lock contention is one of the most insidious performance killers in concurrent systems. A single hot mutex can serialize what should be parallel work, turning your multi-core machine into an effective single-threaded processor. The solution is rarely to remove locks entirely, but to reduce the time spent holding them.",
	"Memory allocation patterns matter enormously at scale. Allocating millions of small objects puts enormous pressure on the garbage collector, leading to stop-the-world pauses that destroy tail latency. Object pooling, arena allocation, and careful struct layout can eliminate most of this pressure.",
	"The CPU cache hierarchy is the single most important hardware detail that software engineers routinely ignore. Cache-friendly data structures can outperform cache-hostile ones by 10x or more on modern hardware. A sequential scan of a slice is often faster than a pointer-chasing tree traversal, even if the tree has better asymptotic complexity.",
	"Distributed consensus is fundamentally about agreeing on a sequence of events across a network where messages can be delayed, reordered, or lost. Raft and Paxos solve this problem with different trade-offs in complexity and understandability. Raft was explicitly designed to be more understandable, and the paper's clarity shows.",
	"Immutability is not just a functional programming concept — it is a powerful tool for building concurrent systems. Immutable data structures can be shared freely across goroutines without any synchronization overhead. The cost is copying on write, but for read-heavy workloads, this trade-off is almost always worth it.",
	"The write-ahead log (WAL) is the foundation of durability in database systems. By writing changes to a sequential log before applying them to the main data structure, databases can recover from crashes without losing committed transactions. Sequential writes are dramatically faster than random writes on both HDDs and SSDs.",
	"Zero-copy I/O techniques like `sendfile` and `splice` on Linux allow data to be transferred between file descriptors without ever copying it into user space. For high-throughput network servers, this can dramatically reduce CPU usage and memory bandwidth consumption.",
	"Bloom filters trade a small probability of false positives for dramatically reduced memory usage. They are ideal for use cases like checking whether a key exists in a large dataset before performing an expensive lookup. The false positive rate is tunable: more bits per element means fewer false positives.",
	"The actor model provides a clean abstraction for concurrent computation. Each actor has its own private state and communicates with other actors exclusively through message passing, eliminating shared mutable state entirely. Erlang's success in building highly available telecom systems proved the model's viability at scale.",
	"Tail call optimization allows recursive functions to run in constant stack space. Languages that guarantee TCO, like Scheme and Elixir, can express certain algorithms more naturally without the risk of stack overflow. Go deliberately does not implement TCO, a design decision that has been debated extensively in the community.",
	"Software transactional memory (STM) brings database-style transactions to in-memory data structures. Operations are executed optimistically and retried if a conflict is detected, providing a composable alternative to locks. Haskell's STM implementation is considered one of the most elegant in any language.",
	"The CAP theorem states that a distributed system can provide at most two of three guarantees: consistency, availability, and partition tolerance. In practice, partition tolerance is non-negotiable, so the real trade-off is between consistency and availability. The PACELC model extends CAP to also consider latency trade-offs in the absence of partitions.",
	"Vectorized execution engines process data in batches rather than one row at a time. By operating on arrays of values, they can leverage SIMD instructions and improve cache utilization, dramatically accelerating analytical queries. DuckDB's vectorized engine achieves remarkable performance on a single machine.",
	"The Linux kernel's epoll mechanism allows a single thread to monitor thousands of file descriptors for I/O readiness. This is the foundation of high-performance event loops used by servers like Nginx and Node.js. The key insight is that most connections are idle most of the time.",
	"Consistent hashing distributes load across a cluster of nodes in a way that minimizes remapping when nodes are added or removed. It is a cornerstone of distributed caching systems like Memcached and Cassandra. Virtual nodes improve load balance at the cost of additional metadata.",
	"Copy-on-write semantics allow multiple readers to share the same data structure without copying it. Only when a writer needs to modify the data is a copy made, making reads extremely cheap in read-heavy workloads. Linux uses COW for process forking, which makes `fork()` nearly instantaneous.",
	"The skip list is a probabilistic data structure that provides O(log n) search, insertion, and deletion. It achieves this through a hierarchy of linked lists, where higher levels skip over more elements. Redis uses skip lists for its sorted set implementation because they are simpler to implement correctly than balanced BSTs.",
	"Compaction in LSM-tree-based storage engines merges sorted runs of data to reclaim space and improve read performance. The trade-off between write amplification, read amplification, and space amplification is the central design challenge of LSM-tree tuning. RocksDB exposes dozens of parameters to control this trade-off.",
	"Service meshes like Istio and Linkerd intercept all network traffic between microservices to provide observability, security, and traffic management without modifying application code. The sidecar proxy pattern is elegant but adds latency and resource overhead that must be carefully measured.",
	"eBPF allows user-defined programs to run safely inside the Linux kernel, enabling powerful observability and networking capabilities without kernel modifications. Tools like Cilium, Pixie, and Falco are built on eBPF and represent a new generation of infrastructure software.",
	"The B-tree is the workhorse of database storage engines. Its high branching factor minimizes the number of disk seeks required to find a key, making it ideal for block-oriented storage. PostgreSQL, MySQL, and SQLite all use B-tree variants as their primary index structure.",
	"Consistent snapshots in distributed databases are achieved through mechanisms like MVCC (Multi-Version Concurrency Control). Each transaction sees a consistent view of the database as of its start time, without blocking readers or writers. PostgreSQL's MVCC implementation is a masterclass in practical database engineering.",
	"The Raft consensus algorithm divides the consensus problem into three relatively independent sub-problems: leader election, log replication, and safety. By tackling these separately, Raft achieves understandability without sacrificing correctness. The original paper includes a formal proof of safety.",
	"Garbage collection pauses are the bane of latency-sensitive applications. Go's concurrent GC has improved dramatically over the years, but sub-millisecond pause times require careful attention to allocation rates. The `GOGC` and `GOMEMLIMIT` environment variables provide coarse-grained control over GC behavior.",
	"Protocol Buffers provide a compact, efficient binary serialization format with strong backward and forward compatibility guarantees. The generated code is fast and the schema evolution rules are well-defined. JSON is more human-readable, but Protobuf is typically 3-10x smaller and faster to serialize.",
	"The two-phase commit protocol (2PC) ensures atomicity across distributed transactions. A coordinator asks all participants to prepare, and only commits if all agree. The protocol is blocking: if the coordinator fails after the prepare phase, participants are stuck until it recovers.",
	"Rate limiting protects services from overload and abuse. Token bucket and leaky bucket algorithms provide smooth rate limiting, while fixed and sliding window counters are simpler to implement. Redis's atomic Lua scripts make it easy to implement distributed rate limiters.",
	"Observability is the ability to understand the internal state of a system from its external outputs. The three pillars — metrics, logs, and traces — each answer different questions. Metrics tell you what is happening, logs tell you why, and traces tell you where.",
}

var tableTemplates = []string{
	`| Operation      | Time Complexity | Space Complexity | Notes                        |
|----------------|----------------|-----------------|------------------------------|
| Insert         | O(log n)        | O(1)            | Amortized over rebalancing   |
| Delete         | O(log n)        | O(1)            | May trigger rebalancing      |
| Search         | O(log n)        | O(1)            | Worst case guaranteed        |
| Min/Max        | O(1)            | O(1)            | With dedicated pointer       |
| Successor      | O(log n)        | O(1)            | In-order traversal step      |
| Range Query    | O(log n + k)    | O(k)            | k = number of results        |`,

	`| Approach           | Throughput   | Latency p50 | Latency p99 | CPU Usage |
|--------------------|-------------|-------------|-------------|-----------|
| Naive (single-core)| 8k req/s    | 12ms        | 95ms        | 100%      |
| Worker Pool (8)    | 62k req/s   | 2ms         | 14ms        | 72%       |
| Lock-Free Queue    | 110k req/s  | 0.9ms       | 6ms         | 58%       |
| Zero-Copy + epoll  | 280k req/s  | 0.3ms       | 1.8ms       | 41%       |
| io_uring           | 420k req/s  | 0.15ms      | 0.9ms       | 33%       |`,

	`| Algorithm      | Best Case      | Average Case   | Worst Case     | Stable | In-Place |
|----------------|----------------|----------------|----------------|--------|----------|
| Quicksort      | O(n log n)     | O(n log n)     | O(n²)          | No     | Yes      |
| Mergesort      | O(n log n)     | O(n log n)     | O(n log n)     | Yes    | No       |
| Heapsort       | O(n log n)     | O(n log n)     | O(n log n)     | No     | Yes      |
| Timsort        | O(n)           | O(n log n)     | O(n log n)     | Yes    | No       |
| Radix Sort     | O(nk)          | O(nk)          | O(nk)          | Yes    | No       |
| Counting Sort  | O(n + k)       | O(n + k)       | O(n + k)       | Yes    | No       |`,

	`| Feature              | PostgreSQL | MySQL   | SQLite  | CockroachDB | DuckDB  |
|----------------------|-----------|---------|---------|-------------|---------|
| ACID Transactions    | ✓         | ✓       | ✓       | ✓           | ✓       |
| MVCC                 | ✓         | ✓       | ✓       | ✓           | ✓       |
| JSON/JSONB           | ✓         | ✓       | ✓       | ✓           | ✓       |
| Full-Text Search     | ✓         | ✓       | ✗       | ✗           | ✗       |
| Distributed          | ✗         | ✗       | ✗       | ✓           | ✗       |
| Columnar Storage     | ✗         | ✗       | ✗       | ✗           | ✓       |
| Parallel Query       | ✓         | ✓       | ✗       | ✓           | ✓       |`,

	`| GC Strategy        | Language    | Pause Time  | Throughput | Tuning Knobs |
|--------------------|-------------|-------------|------------|--------------|
| Stop-the-World     | Early Java  | 100ms–10s   | High       | Heap size    |
| Concurrent Mark    | Go          | <1ms        | Medium     | GOGC, GOMEMLIMIT |
| G1GC               | JVM         | 1–50ms      | High       | Many         |
| ZGC                | JVM         | <1ms        | High       | Region size  |
| Generational       | Python (3.12)| 1–100ms    | Low        | Thresholds   |
| Reference Counting | Swift/Rust  | 0 (no GC)   | Highest    | None         |`,

	`| Consensus Algorithm | Leader Election | Log Replication | Fault Tolerance | Complexity |
|---------------------|----------------|----------------|----------------|------------|
| Paxos               | Complex         | Multi-round    | f < n/2        | Very High  |
| Raft                | Randomized TO   | Single leader  | f < n/2        | Medium     |
| Zab                 | Epoch-based     | Primary-backup | f < n/2        | Medium     |
| Viewstamped Rep.    | View change     | Primary-backup | f < n/2        | Medium     |
| PBFT                | View change     | Three-phase    | f < n/3        | Very High  |`,
}

var chapterThemes = [][]string{
	{
		"Foundations and Mental Models",
		"The Hardware Reality",
		"Measurement Before Optimization",
		"Data Structure Selection",
		"Concurrency Primitives",
		"Memory Allocation Strategies",
		"I/O Patterns and System Calls",
		"Caching at Every Layer",
		"Networking Fundamentals",
		"Distributed Coordination",
		"Observability and Debugging",
		"Production Readiness",
	},
	{
		"Why This Problem Is Hard",
		"Historical Context and Evolution",
		"Core Abstractions",
		"The Implementation",
		"Edge Cases and Failure Modes",
		"Performance Characteristics",
		"Testing and Verification",
		"Operational Concerns",
		"Security Considerations",
		"Scalability Limits",
		"Alternatives and Trade-offs",
		"The Future of the Field",
	},
	{
		"Problem Statement",
		"Prior Art and Related Work",
		"Design Principles",
		"Architecture Overview",
		"The Core Algorithm",
		"Optimizations and Refinements",
		"Benchmarking Methodology",
		"Results and Analysis",
		"Lessons Learned",
		"Open Problems",
		"Practical Guidance",
		"Conclusion",
	},
}

var bulletLists = [][]string{
	{
		"Measure first — a flame graph is worth a thousand assumptions.",
		"Prefer stack allocation over heap allocation in hot paths.",
		"Batch small I/O operations to amortize syscall overhead.",
		"Use read-write locks when reads vastly outnumber writes.",
		"Pre-allocate slices and maps when the final size is known.",
		"Avoid interface boxing in tight loops — it defeats inlining.",
		"Profile with realistic workloads, not synthetic microbenchmarks.",
		"Understand your allocator: `pprof` heap profiles reveal the truth.",
	},
	{
		"Design for failure: every network call can and will fail.",
		"Implement exponential backoff with jitter for retries.",
		"Use circuit breakers to prevent cascading failures.",
		"Set timeouts on every external call — no exceptions.",
		"Make operations idempotent wherever possible.",
		"Log correlation IDs to trace requests across services.",
		"Test failure scenarios in staging before they hit production.",
		"Monitor error rates, not just latency and throughput.",
	},
	{
		"Keep your hot path allocation-free where possible.",
		"Prefer value types over pointer types for small structs.",
		"Use `unsafe.Sizeof` to verify struct layout and padding.",
		"Align frequently-accessed fields to cache line boundaries.",
		"Separate hot and cold data to improve cache utilization.",
		"Use `//go:noescape` and `//go:nosplit` judiciously.",
		"Benchmark with `benchstat` to detect noise in results.",
		"Read the assembly output — the compiler sometimes surprises you.",
	},
	{
		"Understand the CAP theorem before designing your data layer.",
		"Choose eventual consistency only when you can tolerate stale reads.",
		"Use optimistic locking for low-contention update patterns.",
		"Partition your data to avoid cross-shard transactions.",
		"Index selectively — every index slows down writes.",
		"Use connection pooling to amortize TCP handshake costs.",
		"Monitor replication lag in read replicas.",
		"Test your backup and restore procedure regularly.",
	},
}

var numberedLists = [][]string{
	{
		"Establish a baseline: run the system under realistic load and record metrics.",
		"Identify the bottleneck: use profiling tools to find the hot path.",
		"Form a hypothesis: propose a specific change and predict its effect.",
		"Implement the change in isolation, touching as little code as possible.",
		"Measure again: compare against the baseline with statistical rigor.",
		"If improved, commit and document the change. If not, revert and re-hypothesize.",
		"Repeat until the system meets its performance requirements.",
	},
	{
		"Define your SLOs before writing a single line of code.",
		"Instrument your application with structured logging from day one.",
		"Set up distributed tracing before you have more than two services.",
		"Implement health checks and readiness probes for every service.",
		"Write runbooks for every alert before the alert fires in production.",
		"Conduct game days to test your incident response procedures.",
		"Review your on-call rotation and escalation paths quarterly.",
	},
	{
		"Start with the simplest correct implementation.",
		"Write comprehensive tests before optimizing.",
		"Profile to identify the actual bottleneck.",
		"Apply the targeted optimization.",
		"Verify correctness with the existing test suite.",
		"Benchmark to confirm the improvement.",
		"Document the optimization and its rationale.",
	},
}

// ─── Book-length post builder ─────────────────────────────────────────────────

func randomTags(rng *rand.Rand, n int) []string {
	perm := rng.Perm(len(tagPool))
	tags := make([]string, n)
	for i := range tags {
		tags[i] = tagPool[perm[i]]
	}
	return tags
}

func formatDate(t time.Time) string {
	return t.Format("January 2, 2006")
}

func pick[T any](rng *rand.Rand, s []T) T { return s[rng.Intn(len(s))] }

func writeQuote(sb *strings.Builder, rng *rand.Rand) {
	q := pick(rng, quotes)
	fmt.Fprintf(sb, "> %s\n>\n> — %s, *%s*\n\n", q.text, q.author, q.source)
}

func writeSection(sb *strings.Builder, rng *rand.Rand, level int, title string) {
	prefix := strings.Repeat("#", level)
	fmt.Fprintf(sb, "%s %s\n\n", prefix, title)

	// 3–5 paragraphs
	for i := 0; i < 3+rng.Intn(3); i++ {
		sb.WriteString(pick(rng, paragraphs))
		sb.WriteString("\n\n")
	}
}

func buildBookPost(rng *rand.Rand, idx int, date time.Time) string {
	topic := pick(rng, topics)
	tags := randomTags(rng, 4+rng.Intn(4))
	chapters := pick(rng, chapterThemes)
	numChapters := 8 + rng.Intn(5) // 8–12 chapters
	if numChapters > len(chapters) {
		numChapters = len(chapters)
	}

	id := fmt.Sprintf("stress-%04d", idx)
	title := fmt.Sprintf("The Definitive Guide to %s", topic)
	description := fmt.Sprintf(
		"A comprehensive, book-length treatment of %s: from first principles through advanced implementation, "+
			"benchmarking, operational concerns, and the open problems that remain unsolved.",
		strings.ToLower(topic),
	)

	var sb strings.Builder
	sb.Grow(64 * 1024) // pre-allocate 64 KB

	// ── Frontmatter ──────────────────────────────────────────────────────────
	fmt.Fprintf(&sb, "---\nid: %s\ntitle: %s\ndate: %s\ntags: %s\ndescription: %s\n---\n\n",
		id, title, formatDate(date), strings.Join(tags, ", "), description)

	// ── Title & Abstract ─────────────────────────────────────────────────────
	fmt.Fprintf(&sb, "# %s\n\n", title)

	sb.WriteString("## Abstract\n\n")
	for i := 0; i < 3; i++ {
		sb.WriteString(pick(rng, paragraphs))
		sb.WriteString("\n\n")
	}
	writeQuote(&sb, rng)

	// ── Table of Contents ────────────────────────────────────────────────────
	sb.WriteString("## Table of Contents\n\n")
	for i, ch := range chapters[:numChapters] {
		fmt.Fprintf(&sb, "%d. [%s](#chapter-%d)\n", i+1, ch, i+1)
	}
	sb.WriteString("\n---\n\n")

	// ── Chapters ─────────────────────────────────────────────────────────────
	for chIdx, chTitle := range chapters[:numChapters] {
		chNum := chIdx + 1
		fmt.Fprintf(&sb, "## Chapter %d: %s {#chapter-%d}\n\n", chNum, chTitle, chNum)

		// Chapter intro
		for i := 0; i < 2+rng.Intn(2); i++ {
			sb.WriteString(pick(rng, paragraphs))
			sb.WriteString("\n\n")
		}

		// Quote card every other chapter
		if chIdx%2 == 0 {
			writeQuote(&sb, rng)
		}

		// 3–5 sub-sections per chapter
		numSections := 3 + rng.Intn(3)
		subTitles := generateSubTitles(rng, chTitle, numSections)
		for sIdx, subTitle := range subTitles {
			writeSection(&sb, rng, 3, subTitle)

			// Code block in first and last sub-section of each chapter
			if sIdx == 0 || sIdx == numSections-1 {
				snippet := pick(rng, codeSnippets)
				fmt.Fprintf(&sb, "```%s\n%s\n```\n\n", snippet.lang, snippet.code)
				sb.WriteString(pick(rng, paragraphs))
				sb.WriteString("\n\n")
			}

			// Table in the middle sub-section
			if sIdx == numSections/2 {
				sb.WriteString(pick(rng, tableTemplates))
				sb.WriteString("\n\n")
				sb.WriteString(pick(rng, paragraphs))
				sb.WriteString("\n\n")
			}

			// Bullet list in every third sub-section
			if sIdx%3 == 2 {
				items := pick(rng, bulletLists)
				rng.Shuffle(len(items), func(a, b int) { items[a], items[b] = items[b], items[a] })
				for _, item := range items[:4+rng.Intn(4)] {
					fmt.Fprintf(&sb, "- %s\n", item)
				}
				sb.WriteString("\n")
			}

			// Numbered list in every fourth sub-section
			if sIdx%4 == 3 {
				steps := pick(rng, numberedLists)
				for i, step := range steps {
					fmt.Fprintf(&sb, "%d. %s\n", i+1, step)
				}
				sb.WriteString("\n")
			}

			// Nested sub-sub-section occasionally
			if rng.Intn(3) == 0 {
				nestedTitle := fmt.Sprintf("Deep Dive: %s in Practice", subTitle)
				fmt.Fprintf(&sb, "#### %s\n\n", nestedTitle)
				for i := 0; i < 2+rng.Intn(2); i++ {
					sb.WriteString(pick(rng, paragraphs))
					sb.WriteString("\n\n")
				}
				// Extra code block in nested section
				snippet := pick(rng, codeSnippets)
				fmt.Fprintf(&sb, "```%s\n%s\n```\n\n", snippet.lang, snippet.code)
			}
		}

		// Chapter summary box
		fmt.Fprintf(&sb, "> **Chapter %d Summary**\n>\n", chNum)
		summaryItems := pick(rng, bulletLists)
		for _, item := range summaryItems[:3] {
			fmt.Fprintf(&sb, "> - %s\n", item)
		}
		sb.WriteString("\n")

		sb.WriteString("---\n\n")
	}

	// ── Appendices ───────────────────────────────────────────────────────────
	sb.WriteString("## Appendix A: Benchmarking Methodology\n\n")
	for i := 0; i < 3; i++ {
		sb.WriteString(pick(rng, paragraphs))
		sb.WriteString("\n\n")
	}
	snippet := pick(rng, codeSnippets)
	fmt.Fprintf(&sb, "```%s\n%s\n```\n\n", snippet.lang, snippet.code)

	sb.WriteString("## Appendix B: Reference Tables\n\n")
	sb.WriteString(pick(rng, tableTemplates))
	sb.WriteString("\n\n")
	sb.WriteString(pick(rng, tableTemplates))
	sb.WriteString("\n\n")

	sb.WriteString("## Appendix C: Further Reading\n\n")
	readingItems := []string{
		"*The Art of Computer Programming* — Donald Knuth",
		"*Designing Data-Intensive Applications* — Martin Kleppmann",
		"*Systems Performance* — Brendan Gregg",
		"*Database Internals* — Alex Petrov",
		"*The Linux Programming Interface* — Michael Kerrisk",
		"*Programming Language Pragmatics* — Michael L. Scott",
		"*Computer Networks* — Andrew Tanenbaum",
		"*Operating Systems: Three Easy Pieces* — Arpaci-Dusseau",
		"*Release It!* — Michael T. Nygard",
		"*Site Reliability Engineering* — Google SRE Team",
	}
	rng.Shuffle(len(readingItems), func(a, b int) { readingItems[a], readingItems[b] = readingItems[b], readingItems[a] })
	for _, item := range readingItems[:5+rng.Intn(5)] {
		fmt.Fprintf(&sb, "- %s\n", item)
	}
	sb.WriteString("\n")

	// ── Closing quote ─────────────────────────────────────────────────────────
	writeQuote(&sb, rng)

	// ── Final paragraph ───────────────────────────────────────────────────────
	sb.WriteString(pick(rng, paragraphs))
	sb.WriteString("\n")

	return sb.String()
}

// generateSubTitles produces contextual sub-section titles for a chapter.
func generateSubTitles(rng *rand.Rand, chapterTitle string, n int) []string {
	pools := [][]string{
		{"Overview", "Core Concepts", "Implementation Details", "Performance Implications", "Common Pitfalls"},
		{"Motivation", "The Naive Approach", "The Optimized Approach", "Benchmarks", "Takeaways"},
		{"Background", "The Algorithm", "Correctness Proof", "Complexity Analysis", "Real-World Usage"},
		{"Problem Definition", "Design Space", "Our Approach", "Trade-offs", "Evaluation"},
		{"History", "State of the Art", "Key Innovations", "Limitations", "Future Directions"},
	}
	pool := pick(rng, pools)
	rng.Shuffle(len(pool), func(a, b int) { pool[a], pool[b] = pool[b], pool[a] })
	if n > len(pool) {
		n = len(pool)
	}
	return pool[:n]
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	const count = 1000
	outputDir := filepath.Clean("../../blogs/stress")

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir: %v\n", err)
		os.Exit(1)
	}

	rng := rand.New(rand.NewSource(42))

	// Spread dates over the last 3 years
	baseDate := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	totalDays := 365 * 3

	start := time.Now()
	totalBytes := 0

	for i := 1; i <= count; i++ {
		daysOffset := rng.Intn(totalDays)
		date := baseDate.AddDate(0, 0, daysOffset)

		content := buildBookPost(rng, i, date)
		totalBytes += len(content)

		filename := filepath.Join(outputDir, fmt.Sprintf("stress-%04d.md", i))
		if err := os.WriteFile(filename, []byte(content), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", filename, err)
			os.Exit(1)
		}

		if i%100 == 0 {
			fmt.Printf("  [%d/%d] %.1f MB written so far...\n", i, count, float64(totalBytes)/1e6)
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\nGenerated %d book-length posts in %v\n", count, elapsed)
	fmt.Printf("Total markdown: %.1f MB (avg %.1f KB/post)\n",
		float64(totalBytes)/1e6, float64(totalBytes)/float64(count)/1024)
	fmt.Printf("Output: %s\n", outputDir)
}
