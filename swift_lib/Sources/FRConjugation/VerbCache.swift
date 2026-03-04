// VerbCache.swift — Thread-safe LRU cache keyed by verb infinitive.
//
// Each cache entry stores *all* predicted forms for a single verb,
// across every voice/mode/tense/person combination.  The cache size
// is measured in **verbs** — not individual forms.

import Foundation

// MARK: - Cache Entry

/// All cached predictions for a single verb infinitive.
///
/// This is an internal type — callers interact with the cache
/// exclusively through ``Conjugator``.
struct VerbCacheEntry {
    /// Predicted forms: key = "voice|mode|tense|person", value = conjugated form.
    var forms: [String: String] = [:]
}

// MARK: - LRU Cache

/// A fixed-capacity, least-recently-used cache keyed by verb infinitive.
///
/// **Thread safety:** This type is *not* internally synchronised.
/// The owning ``Conjugator`` must hold its lock before calling any
/// mutating method.
///
/// **Capacity semantics:** One cache slot = one verb.  All forms
/// (across all voices, modes, tenses, persons) for that verb share
/// the same slot.
final class VerbCache {

    /// Maximum number of verbs to cache.  `0` disables caching entirely.
    let capacity: Int

    /// Doubly-linked-list node for O(1) move-to-front / eviction.
    private final class Node {
        let key: String
        var entry: VerbCacheEntry
        var prev: Node?
        var next: Node?

        init(key: String, entry: VerbCacheEntry) {
            self.key = key
            self.entry = entry
        }
    }

    /// Hash map for O(1) lookup by verb.
    private var map: [String: Node]

    /// Sentinel head (most recently used) and tail (least recently used).
    private let head = Node(key: "", entry: VerbCacheEntry())
    private let tail = Node(key: "", entry: VerbCacheEntry())

    // MARK: - Init

    /// Create a cache with the given verb capacity.
    ///
    /// - Parameter capacity: Maximum number of verbs to keep cached.
    ///   Pass `0` to disable caching entirely.
    init(capacity: Int) {
        precondition(capacity >= 0, "Cache capacity must be non-negative")
        self.capacity = capacity
        self.map = [String: Node](minimumCapacity: capacity)
        head.next = tail
        tail.prev = head
    }

    // MARK: - Lookup

    /// The number of verbs currently in the cache.
    var count: Int { map.count }

    /// Look up a cached form.
    ///
    /// - Parameters:
    ///   - verb: The infinitive.
    ///   - formKey: Composite key `"voice|mode|tense|person"`.
    /// - Returns: The cached form string, or `nil` on a miss.
    func get(verb: String, formKey: String) -> String? {
        guard capacity > 0, let node = map[verb] else { return nil }
        moveToFront(node)
        return node.entry.forms[formKey]
    }

    /// Whether the cache contains *any* entry for this verb (even if
    /// the specific form hasn't been cached yet).
    func containsVerb(_ verb: String) -> Bool {
        guard capacity > 0 else { return false }
        return map[verb] != nil
    }

    // MARK: - Insertion

    /// Store a single form in the cache, creating a verb entry if needed.
    ///
    /// If adding a new verb would exceed capacity, the least-recently-used
    /// verb (and all its forms) is evicted.
    ///
    /// - Parameters:
    ///   - verb: The infinitive.
    ///   - formKey: Composite key `"voice|mode|tense|person"`.
    ///   - value: The conjugated form string.
    func set(verb: String, formKey: String, value: String) {
        guard capacity > 0 else { return }

        if let node = map[verb] {
            // Verb already cached — update the form and promote.
            node.entry.forms[formKey] = value
            moveToFront(node)
        } else {
            // New verb — evict LRU if full.
            if map.count >= capacity {
                evictLRU()
            }
            let node = Node(key: verb, entry: VerbCacheEntry(forms: [formKey: value]))
            map[verb] = node
            insertAfterHead(node)
        }
    }

    // MARK: - Eviction

    /// Remove all cached entries.
    func removeAll() {
        map.removeAll(keepingCapacity: true)
        head.next = tail
        tail.prev = head
    }

    // MARK: - Linked-List Helpers

    private func moveToFront(_ node: Node) {
        detach(node)
        insertAfterHead(node)
    }

    private func insertAfterHead(_ node: Node) {
        node.prev = head
        node.next = head.next
        head.next?.prev = node
        head.next = node
    }

    private func detach(_ node: Node) {
        node.prev?.next = node.next
        node.next?.prev = node.prev
        node.prev = nil
        node.next = nil
    }

    private func evictLRU() {
        guard let lru = tail.prev, lru !== head else { return }
        detach(lru)
        map.removeValue(forKey: lru.key)
    }
}
