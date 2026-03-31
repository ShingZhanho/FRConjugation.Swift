// Conjugator.swift -- Idiomatic Swift interface to the French verb conjugation model.

import Foundation

/// A French verb conjugation engine backed by a character-level seq2seq
/// neural network with Bahdanau attention.
///
/// ## Overview
///
/// `Conjugator` loads a pre-trained model from disk and exposes a fully
/// typed Swift API for conjugating French verbs.  The model covers all
/// five grammatical voices (active-avoir, active-être, active, passive,
/// pronominal) with 13 gender-explicit person keys (including `3sn`
/// for the pronoun *on* in reciprocal verbs).
///
/// All conjugation forms -- including compound tenses and participles --
/// are predicted directly by the neural model (with an exception table
/// for the tiny fraction it gets wrong).
///
/// ```swift
/// let c = try Conjugator()
///
/// // Single form
/// c.conjugate("aller", voice: .activeEtre, mode: .indicatif,
///             tense: .present, person: .firstSingularMasculine)
/// // -> "vais"
///
/// // All persons for a tense
/// let forms = c.conjugate("finir", voice: .activeAvoir,
///                         mode: .indicatif, tense: .imparfait)
/// // -> [.firstSingularMasculine: "finissais", ...]
///
/// // Discover valid voices for a verb
/// c.voices("aller")
/// // -> [.activeEtre, .pronominal]
/// ```
///
/// ## Caching
///
/// Each `Conjugator` instance maintains an internal **LRU cache** that
/// stores previously-predicted forms indexed by verb infinitive.  The
/// cache size is measured in **verbs** -- all forms for the same verb
/// share a single cache slot.
///
/// ```swift
/// // Default: 64-verb cache
/// let fr = try Conjugator()
///
/// // Custom cache size (256 verbs)
/// let fr = try Conjugator(cacheSize: 256)
///
/// // Disable caching entirely
/// let fr = try Conjugator(cacheSize: 0)
///
/// // Configure the shared singleton's cache (must be first call)
/// let fr = Conjugator.getShared(cacheSize: 128)
/// ```
///
/// ## Thread Safety
///
/// `Conjugator` is fully thread-safe.  All public methods synchronise
/// access to the underlying inference engine and cache via an internal
/// lock, and the type conforms to `Sendable` so it can be shared
/// freely across concurrency domains.  A shared singleton is available
/// via ``getShared()`` for convenience.
public final class Conjugator: @unchecked Sendable {

    // MARK: - Shared Instance

    /// The default cache capacity used when none is specified.
    public static let defaultCacheSize = 64

    private static var _shared: Conjugator?
    private static let _sharedLock = NSLock()

    /// Returns the shared singleton backed by bundled model resources.
    ///
    /// The singleton is lazily initialised on the **first call**.  You
    /// may optionally pass `cacheSize` on the *first* call to configure
    /// the cache; subsequent calls ignore the parameter.
    ///
    ///     // First call -- sets cache to 128 verbs:
    ///     let fr = Conjugator.getShared(cacheSize: 128)
    ///
    ///     // Later calls -- returns the same instance (cacheSize ignored):
    ///     let fr = Conjugator.getShared()
    ///
    /// - Parameter cacheSize: Maximum number of verbs to cache.
    ///   Only honoured on the first call.  Defaults to
    ///   ``defaultCacheSize`` (64).
    /// - Returns: The shared `Conjugator` instance.
    public static func getShared(cacheSize: Int = defaultCacheSize) -> Conjugator {
        _sharedLock.lock()
        defer { _sharedLock.unlock() }
        if let existing = _shared { return existing }
        let instance = try! Conjugator(cacheSize: cacheSize)
        _shared = instance
        return instance
    }

    /// Returns the shared singleton asynchronously.
    ///
    /// - Parameter cacheSize: Maximum number of verbs to cache.
    ///   Only honoured on the first call.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public static func getShared(cacheSize: Int = defaultCacheSize) async throws -> Conjugator {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(returning: getShared(cacheSize: cacheSize))
            }
        }
    }

    /// Reset the shared singleton.  Internal -- used by tests to ensure
    /// a fresh state between test methods.
    static func _resetShared() {
        _sharedLock.lock()
        defer { _sharedLock.unlock() }
        _shared = nil
    }

    // MARK: - Private State

    private let engine: InferenceEngine
    private let lock = NSLock()
    private let cache: VerbCache

    // MARK: - Initialization

    /// Load the conjugation model from a directory path.
    ///
    /// - Parameters:
    ///   - path: Absolute path to the directory containing
    ///     `model.json` and `weights.bin`.
    ///   - cacheSize: Maximum number of verbs to keep in the LRU
    ///     cache.  Pass `0` to disable caching.  Defaults to
    ///     ``defaultCacheSize`` (64).
    /// - Throws: ``ConjugationError/modelLoadFailed(path:)`` if loading fails.
    public init(modelDirectory path: String, cacheSize: Int = defaultCacheSize) throws {
        let url = URL(fileURLWithPath: path, isDirectory: true)
        self.engine = try InferenceEngine(modelDirectory: url)
        self.cache = VerbCache(capacity: cacheSize)
    }

    /// Load the conjugation model from a directory URL.
    ///
    /// - Parameters:
    ///   - url: File URL to the model directory.
    ///   - cacheSize: Maximum number of verbs to cache (default: ``defaultCacheSize``).
    public convenience init(modelDirectory url: URL, cacheSize: Int = defaultCacheSize) throws {
        try self.init(modelDirectory: url.path, cacheSize: cacheSize)
    }

    /// Load the conjugation model from the bundled resources.
    ///
    /// - Parameter cacheSize: Maximum number of verbs to cache
    ///   (default: ``defaultCacheSize``).
    public convenience init(cacheSize: Int = defaultCacheSize) throws {
        let bundle = Self.resourceBundle
        guard let jsonURL = bundle.url(forResource: "model", withExtension: "json"),
              let _ = bundle.url(forResource: "weights", withExtension: "bin") else {
            throw ConjugationError.modelLoadFailed(path: "bundled resources")
        }
        let dir = jsonURL.deletingLastPathComponent()
        try self.init(modelDirectory: dir, cacheSize: cacheSize)
    }

    private static var resourceBundle: Bundle {
        #if SWIFT_PACKAGE
        return Bundle.module
        #else
        return Bundle(for: Conjugator.self)
        #endif
    }

    // MARK: - Properties

    /// The number of verbs the model recognises.
    ///
    /// Homonym groups (e.g. `ressortir_1`, `ressortir_2`) count as a single verb.
    public var verbCount: Int {
        lock.lock()
        defer { lock.unlock() }
        // Each homonym group contributes 1 base name instead of N suffixed entries.
        var count = engine.knownVerbs.count
        for (_, indices) in engine.homonymMap {
            count -= indices.count
            count += 1
        }
        return count
    }

    /// A sorted list of all verb infinitives known to the model.
    ///
    /// Homonym-suffixed entries (e.g. `ressortir_1`, `ressortir_2`) are
    /// collapsed into their base infinitive (`ressortir`).
    ///
    ///     conjugator.allVerbs
    ///     // ["abaisser", "abandonner", "abasourdir", ...]
    public var allVerbs: [String] {
        lock.lock()
        defer { lock.unlock() }
        var verbs = engine.knownVerbs
        for (base, indices) in engine.homonymMap {
            for idx in indices {
                verbs.remove("\(base)_\(idx)")
            }
            verbs.insert(base)
        }
        return verbs.sorted { a, b in
            let ak = Conjugator.sortKey(a)
            let bk = Conjugator.sortKey(b)
            return ak == bk ? a < b : ak < bk
        }
    }

    /// Produce a diacritics-insensitive, ligature-expanded sort key.
    private static func sortKey(_ s: String) -> String {
        s.replacingOccurrences(of: "œ", with: "oe")
         .replacingOccurrences(of: "Œ", with: "OE")
         .replacingOccurrences(of: "æ", with: "ae")
         .replacingOccurrences(of: "Æ", with: "AE")
         .folding(options: [.diacriticInsensitive, .widthInsensitive],
                  locale: Locale(identifier: "fr"))
    }

    /// The maximum number of verbs the LRU cache can hold.
    ///
    /// Returns `0` if caching is disabled.
    public var cacheCapacity: Int {
        cache.capacity
    }

    /// The number of verbs currently held in the cache.
    public var cacheCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    /// Remove all entries from the prediction cache.
    public func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
    }

    // MARK: - Queries

    /// Whether the model recognises this infinitive.
    ///
    /// For verbs with homonyms (e.g. *ressortir*), returns `true` for
    /// the base name even though the internal keys are suffixed
    /// (*ressortir_1*, *ressortir_2*).
    public func hasVerb(_ infinitive: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return engine.knownVerbs.contains(infinitive)
            || engine.homonymMap[infinitive] != nil
    }

    /// Whether a verb beginning with *h* is h-aspiré (no elision/liaison).
    public func isHAspire(_ infinitive: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return engine.hAspire.contains(infinitive)
    }

    /// Whether a verb has 1990 reform spelling changes.
    public func is1990Reform(_ infinitive: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return engine.reform1990Verbs.contains(infinitive)
    }

    /// Return the 1990 reform variant for a verb, or `nil`.
    public func reformVariante(_ infinitive: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return engine.reformVariantes[infinitive]
    }

    // MARK: - Homonym Queries

    /// Whether the given base infinitive has multiple homonym entries.
    ///
    ///     conjugator.hasHomonyms("ressortir")  // true
    ///     conjugator.hasHomonyms("parler")     // false
    public func hasHomonyms(_ infinitive: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return engine.homonymMap[infinitive] != nil
    }

    /// The number of homonym entries for a verb, or 1 if it has no homonyms.
    ///
    ///     conjugator.homonymCount("ressortir")  // 2
    ///     conjugator.homonymCount("parler")     // 1
    public func homonymCount(_ infinitive: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return engine.homonymMap[infinitive]?.count ?? 1
    }

    /// The available homonym indices for a verb, or an empty array if none.
    ///
    ///     conjugator.homonymIndices("ressortir")  // [1, 2]
    ///     conjugator.homonymIndices("parler")     // []
    public func homonymIndices(_ infinitive: String) -> [Int] {
        lock.lock()
        defer { lock.unlock() }
        return engine.homonymMap[infinitive] ?? []
    }

    // MARK: - Homonym Key Resolution

    /// Resolve a user-facing infinitive + optional homonym index to the
    /// internal DB key used for lookups.
    ///
    /// - If the infinitive is directly known (e.g. "parler"), returns it as-is.
    /// - If the infinitive has homonyms and `homonymIndex` is nil, defaults to
    ///   the first index (1).
    /// - If `homonymIndex` is provided, returns "`infinitive`_`index`".
    ///
    /// **Must be called while `lock` is held.**
    private func resolveKey(_ infinitive: String, homonymIndex: Int?) -> String {
        // Direct match — no homonyms
        if engine.knownVerbs.contains(infinitive) {
            return infinitive
        }
        // Homonym verb — resolve to suffixed key
        if let indices = engine.homonymMap[infinitive] {
            let idx = homonymIndex ?? indices.first ?? 1
            return "\(infinitive)_\(idx)"
        }
        // Unknown verb — return as-is (will fail downstream)
        return infinitive
    }

    // MARK: - Structure Queries

    /// List available voices for a verb.
    ///
    ///     conjugator.voices("aller")
    ///     // -> [.activeEtre, .pronominal]
    public func voices(_ infinitive: String, homonymIndex: Int? = nil) -> [Voice] {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let struct_ = engine.verbStructure[key] else { return [] }
        return struct_.keys.sorted().compactMap { Voice(rawValue: $0) }
    }

    /// List available modes for a verb in a given voice.
    public func modes(_ infinitive: String, voice: Voice, homonymIndex: Int? = nil) -> [Mode] {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let voiceStruct = engine.verbStructure[key]?[voice.rawValue] else { return [] }
        return voiceStruct.keys.sorted().compactMap { Mode(rawValue: $0) }
    }

    /// List available tenses for a verb in a given voice and mode.
    public func tenses(_ infinitive: String, voice: Voice, mode: Mode, homonymIndex: Int? = nil) -> [Tense] {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let modeStruct = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue] else { return [] }
        return modeStruct.keys.sorted().compactMap { Tense(rawValue: $0) }
    }

    /// List available persons for a verb in a given voice, mode and tense.
    public func persons(_ infinitive: String, voice: Voice, mode: Mode, tense: Tense, homonymIndex: Int? = nil) -> [Person] {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue] else { return [] }
        return persons.compactMap { Person(rawValue: $0) }
    }

    // MARK: - Private Cache Helper

    /// Predict a single form, reading from / writing to the LRU cache.
    ///
    /// **Must be called while `lock` is held.**
    private func cachedPredict(
        infinitive: String,
        voice: String,
        mode: String,
        tense: String,
        person: String
    ) -> String? {
        let formKey = "\(voice)|\(mode)|\(tense)|\(person)"
        if let hit = cache.get(verb: infinitive, formKey: formKey) {
            return hit
        }
        guard let result = engine.predict(
            infinitive: infinitive,
            voice: voice,
            mode: mode,
            tense: tense,
            person: person
        ) else {
            return nil
        }
        cache.set(verb: infinitive, formKey: formKey, value: result)
        return result
    }

    // MARK: - Form Splitting

    /// Extract the primary (first) form from a raw prediction that may
    /// contain semicolon-separated alternatives (e.g. "abrégerai;abrègerai").
    private static func primaryForm(_ raw: String) -> String {
        if let idx = raw.firstIndex(of: ";") {
            return String(raw[raw.startIndex..<idx])
        }
        return raw
    }

    /// Extract the alternative (second) form from a raw prediction,
    /// falling back to the primary form when no alternative exists.
    private static func alternativeForm(_ raw: String) -> String {
        if let idx = raw.firstIndex(of: ";") {
            return String(raw[raw.index(after: idx)...])
        }
        return raw
    }

    // MARK: - Conjugation (Single Form)

    /// Conjugate a single form.
    ///
    /// When the model predicts multiple spelling variants (e.g.
    /// *abrégerai* / *abrègerai*), this method returns the **primary**
    /// (first) variant.  Use ``conjugateAlternative(_:voice:mode:tense:person:)``
    /// to obtain the alternative spelling.
    ///
    ///     conjugator.conjugate("aller", voice: .activeEtre,
    ///         mode: .indicatif, tense: .present,
    ///         person: .firstSingularMasculine)
    ///     // -> "vais"
    ///
    /// - Returns: The conjugated form, or `nil` if the combination is
    ///   invalid or the verb is unknown.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) -> String? {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        // Validate against verb_structure
        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return nil
        }

        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        ) else { return nil }
        return Self.primaryForm(raw)
    }

    /// Return the alternative spelling for a single conjugated form.
    ///
    /// If the form has two spelling variants (e.g. *abrégerai* /
    /// *abrègerai*), this returns the **second** variant.  When only
    /// one spelling exists, it behaves identically to
    /// ``conjugate(_:voice:mode:tense:person:)``.
    ///
    ///     conjugator.conjugateAlternative("abréger",
    ///         voice: .activeAvoir, mode: .indicatif,
    ///         tense: .futurSimple, person: .firstSingularMasculine)
    ///     // -> "abrègerai"
    ///
    /// - Returns: The alternative (or only) conjugated form, or `nil`
    ///   if the combination is invalid.
    public func conjugateAlternative(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) -> String? {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return nil
        }

        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        ) else { return nil }
        return Self.alternativeForm(raw)
    }

    /// Whether the given conjugation has an alternative spelling variant.
    ///
    ///     conjugator.hasAlternativeForm("abréger",
    ///         voice: .activeAvoir, mode: .indicatif,
    ///         tense: .futurSimple, person: .firstSingularMasculine)
    ///     // -> true
    ///
    /// - Returns: `true` if there are two spelling variants, `false` otherwise.
    public func hasAlternativeForm(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return false
        }

        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        ) else { return false }
        return raw.contains(";")
    }

    /// Conjugate all persons for a given voice, mode and tense.
    ///
    /// - Returns: A dictionary mapping each valid person to its form.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        homonymIndex: Int? = nil
    ) -> [Person: String] {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let personKeys = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue] else {
            return [:]
        }

        var result = [Person: String](minimumCapacity: personKeys.count)
        for pKey in personKeys {
            guard let person = Person(rawValue: pKey) else { continue }
            if let form = cachedPredict(
                infinitive: key,
                voice: voice.rawValue,
                mode: mode.rawValue,
                tense: tense.rawValue,
                person: pKey
            ) {
                result[person] = Self.primaryForm(form)
            }
        }
        return result
    }

    /// Conjugate all tenses and persons for a voice and mode.
    ///
    /// - Returns: A nested dictionary: tense -> person -> form.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        homonymIndex: Int? = nil
    ) -> [Tense: [Person: String]] {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let modeStruct = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue] else {
            return [:]
        }

        var result = [Tense: [Person: String]]()
        for (tenseKey, personKeys) in modeStruct {
            guard let tense = Tense(rawValue: tenseKey) else { continue }
            var tenseResult = [Person: String](minimumCapacity: personKeys.count)
            for pKey in personKeys {
                guard let person = Person(rawValue: pKey) else { continue }
                if let form = cachedPredict(
                    infinitive: key,
                    voice: voice.rawValue,
                    mode: mode.rawValue,
                    tense: tenseKey,
                    person: pKey
                ) {
                    tenseResult[person] = Self.primaryForm(form)
                }
            }
            if !tenseResult.isEmpty {
                result[tense] = tenseResult
            }
        }
        return result
    }

    /// Conjugate all modes, tenses and persons for a voice.
    ///
    /// - Returns: A nested dictionary: mode -> tense -> person -> form.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        homonymIndex: Int? = nil
    ) -> [Mode: [Tense: [Person: String]]] {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let voiceStruct = engine.verbStructure[key]?[voice.rawValue] else {
            return [:]
        }

        var result = [Mode: [Tense: [Person: String]]]()
        for (modeKey, modeTenses) in voiceStruct {
            guard let mode = Mode(rawValue: modeKey) else { continue }
            var modeResult = [Tense: [Person: String]]()
            for (tenseKey, personKeys) in modeTenses {
                guard let tense = Tense(rawValue: tenseKey) else { continue }
                var tenseResult = [Person: String](minimumCapacity: personKeys.count)
                for pKey in personKeys {
                    guard let person = Person(rawValue: pKey) else { continue }
                    if let form = cachedPredict(
                        infinitive: key,
                        voice: voice.rawValue,
                        mode: modeKey,
                        tense: tenseKey,
                        person: pKey
                    ) {
                        tenseResult[person] = Self.primaryForm(form)
                    }
                }
                if !tenseResult.isEmpty {
                    modeResult[tense] = tenseResult
                }
            }
            if !modeResult.isEmpty {
                result[mode] = modeResult
            }
        }
        return result
    }

    /// Conjugate all voices, modes, tenses and persons for a verb.
    ///
    /// - Returns: A nested dictionary: voice -> mode -> tense -> person -> form,
    ///   or `nil` if the verb is unknown.
    public func conjugate(
        _ infinitive: String,
        homonymIndex: Int? = nil
    ) -> [Voice: [Mode: [Tense: [Person: String]]]]? {
        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let verbStruct = engine.verbStructure[key] else {
            return nil
        }

        var result = [Voice: [Mode: [Tense: [Person: String]]]]()
        for (voiceKey, voiceModes) in verbStruct {
            guard let voice = Voice(rawValue: voiceKey) else { continue }
            var voiceResult = [Mode: [Tense: [Person: String]]]()
            for (modeKey, modeTenses) in voiceModes {
                guard let mode = Mode(rawValue: modeKey) else { continue }
                var modeResult = [Tense: [Person: String]]()
                for (tenseKey, personKeys) in modeTenses {
                    guard let tense = Tense(rawValue: tenseKey) else { continue }
                    var tenseResult = [Person: String](minimumCapacity: personKeys.count)
                    for pKey in personKeys {
                        guard let person = Person(rawValue: pKey) else { continue }
                        if let form = cachedPredict(
                            infinitive: key,
                            voice: voiceKey,
                            mode: modeKey,
                            tense: tenseKey,
                            person: pKey
                        ) {
                            tenseResult[person] = Self.primaryForm(form)
                        }
                    }
                    if !tenseResult.isEmpty {
                        modeResult[tense] = tenseResult
                    }
                }
                if !modeResult.isEmpty {
                    voiceResult[mode] = modeResult
                }
            }
            if !voiceResult.isEmpty {
                result[voice] = voiceResult
            }
        }
        return result.isEmpty ? nil : result
    }

    // MARK: - Participles

    /// Get a single participle form.
    ///
    ///     conjugator.participle("partir", voice: .activeEtre,
    ///                           tense: .passeFemininPluriel)
    ///     // -> "parties"
    ///
    ///     conjugator.participle("parler", voice: .activeAvoir,
    ///                           tense: .present)
    ///     // -> "parlant"
    ///
    /// - Parameters:
    ///   - infinitive: The verb infinitive.
    ///   - voice: The grammatical voice.
    ///   - tense: Which participle form to retrieve (e.g. `.present`,
    ///     `.passeMasculinSingulier`, `.passeFemininPluriel`).
    /// - Returns: The participle string, or `nil` if unavailable.
    /// Get a single participle form.
    ///
    /// When variants exist, returns the **primary** form.
    /// Use ``participleAlternative(_:voice:tense:)`` for the alternative.
    public func participle(
        _ infinitive: String,
        voice: Voice,
        tense: Tense,
        homonymIndex: Int? = nil
    ) -> String? {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let persons = engine.verbStructure[key]?[voice.rawValue]?["participe"]?[tense.rawValue],
              persons.contains("-") else {
            return nil
        }
        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: "participe",
            tense: tense.rawValue,
            person: "-"
        ) else { return nil }
        return Self.primaryForm(raw)
    }

    /// Return the alternative spelling for a single participle form.
    ///
    /// If only one spelling exists, behaves identically to
    /// ``participle(_:voice:tense:)``.
    public func participleAlternative(
        _ infinitive: String,
        voice: Voice,
        tense: Tense,
        homonymIndex: Int? = nil
    ) -> String? {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let persons = engine.verbStructure[key]?[voice.rawValue]?["participe"]?[tense.rawValue],
              persons.contains("-") else {
            return nil
        }
        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: "participe",
            tense: tense.rawValue,
            person: "-"
        ) else { return nil }
        return Self.alternativeForm(raw)
    }

    /// Whether the given participle has an alternative spelling variant.
    public func hasAlternativeParticiple(
        _ infinitive: String,
        voice: Voice,
        tense: Tense,
        homonymIndex: Int? = nil
    ) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let persons = engine.verbStructure[key]?[voice.rawValue]?["participe"]?[tense.rawValue],
              persons.contains("-") else {
            return false
        }
        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: "participe",
            tense: tense.rawValue,
            person: "-"
        ) else { return false }
        return raw.contains(";")
    }

    // MARK: - Pronoun

    /// French vowels used for elision checks.
    private static let frenchVowels: Set<Character> = ["a", "e", "i", "o", "u",
                                                        "\u{00E0}", "\u{00E2}", "\u{00E9}", "\u{00E8}", "\u{00EA}", "\u{00EB}",
                                                        "\u{00EE}", "\u{00EF}", "\u{00F4}", "\u{00F9}", "\u{00FB}", "\u{00FC}",
                                                        "\u{0153}", "\u{00E6}", "y"]

    /// Whether a conjugated form starts with a sound that triggers elision.
    ///
    /// **Must be called while `lock` is held.**
    private func formStartsWithVowelSound(_ form: String, infinitive: String) -> Bool {
        guard let first = form.first else { return false }
        let lower = Character(first.lowercased())
        if lower == "h" {
            return !engine.hAspire.contains(infinitive)
        }
        return Self.frenchVowels.contains(lower)
    }

    /// Return the contextual subject pronoun for a conjugated form.
    ///
    /// The pronoun accounts for elision ("je" becomes "j'" before a
    /// vowel sound or h-muet) and prepends *que* / *qu'* for the
    /// subjonctif mood.
    ///
    ///     conjugator.getPronoun("aimer", voice: .activeAvoir,
    ///         mode: .indicatif, tense: .present,
    ///         person: .firstSingularMasculine)
    ///     // -> "j'"
    ///
    ///     conjugator.getPronoun("parler", voice: .activeAvoir,
    ///         mode: .subjonctif, tense: .present,
    ///         person: .thirdSingularMasculine)
    ///     // -> "qu'il "
    ///
    /// - Returns: The pronoun string (with trailing space or apostrophe),
    ///   or `nil` for imperatif, participe, unknown verbs, or invalid
    ///   combinations.
    public func getPronoun(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) -> String? {
        // No subject pronoun for imperatif or participe
        guard mode != .imperatif, mode != .participe else { return nil }

        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        // Validate combination and get the conjugated form
        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return nil
        }
        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        ) else { return nil }

        let form = Self.primaryForm(raw)
        return pronounString(person: person, mode: mode, form: form, infinitive: infinitive)
    }

    /// Build the pronoun string for a given person, mode, and conjugated form.
    ///
    /// **Must be called while `lock` is held.**
    private func pronounString(person: Person, mode: Mode, form: String, infinitive: String) -> String {
        let base = person.pronoun
        let vowelSound = formStartsWithVowelSound(form, infinitive: infinitive)

        // Determine the subject pronoun (with possible elision)
        let subject: String
        if (person == .firstSingularMasculine || person == .firstSingularFeminine) && vowelSound {
            subject = "j'"
        } else {
            subject = base + " "
        }

        // Subjonctif: prepend "que" / "qu'"
        if mode == .subjonctif {
            let pronounStartsWithVowel: Bool
            switch person {
            case .thirdSingularMasculine, .thirdSingularFeminine,
                 .thirdSingularNeutral,
                 .thirdPluralMasculine, .thirdPluralFeminine:
                pronounStartsWithVowel = true    // il, elle, on, ils, elles
            default:
                pronounStartsWithVowel = false
            }

            if pronounStartsWithVowel {
                return "qu'" + subject
            }
            return "que " + subject
        }

        return subject
    }

    /// Return the conjugated form prefixed with its contextual subject pronoun.
    ///
    /// Combines ``getPronoun(_:voice:mode:tense:person:)`` and
    /// ``conjugate(_:voice:mode:tense:person:)`` into a single call.
    /// For modes with no subject pronoun (imperatif, participe), the
    /// bare conjugated form is returned.
    ///
    ///     conjugator.conjugateWithPronoun("aimer", voice: .activeAvoir,
    ///         mode: .indicatif, tense: .present,
    ///         person: .firstSingularMasculine)
    ///     // -> "j'aime"
    ///
    ///     conjugator.conjugateWithPronoun("parler", voice: .activeAvoir,
    ///         mode: .imperatif, tense: .present,
    ///         person: .secondSingularMasculine)
    ///     // -> "parle"
    ///
    /// - Returns: The pronoun + form string, or `nil` if the verb is
    ///   unknown or the combination is invalid.
    public func conjugateWithPronoun(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) -> String? {
        // Modes with no subject pronoun -- return bare form
        if mode == .imperatif || mode == .participe {
            return conjugate(infinitive, voice: voice, mode: mode,
                             tense: tense, person: person, homonymIndex: homonymIndex)
        }

        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return nil
        }
        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        ) else { return nil }

        let form = Self.primaryForm(raw)
        let pronoun = pronounString(person: person, mode: mode, form: form, infinitive: infinitive)
        return pronoun + form
    }

    /// Return the alternative conjugated form prefixed with its contextual
    /// subject pronoun.
    ///
    /// Combines ``getPronoun(_:voice:mode:tense:person:)`` and
    /// ``conjugateAlternative(_:voice:mode:tense:person:)`` into a single call.
    /// For modes with no subject pronoun (imperatif, participe), the
    /// bare alternative form is returned.
    ///
    ///     conjugator.conjugateAlternativeWithPronoun("abr\u{00E9}ger",
    ///         voice: .activeAvoir, mode: .indicatif,
    ///         tense: .futurSimple, person: .firstSingularMasculine)
    ///     // -> "j'abr\u{00E8}gerai"
    ///
    /// - Returns: The pronoun + alternative form string, or `nil` if the
    ///   verb is unknown or the combination is invalid.
    public func conjugateAlternativeWithPronoun(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) -> String? {
        if mode == .imperatif || mode == .participe {
            return conjugateAlternative(infinitive, voice: voice, mode: mode,
                                        tense: tense, person: person, homonymIndex: homonymIndex)
        }

        lock.lock()
        defer { lock.unlock() }

        let key = resolveKey(infinitive, homonymIndex: homonymIndex)

        guard let persons = engine.verbStructure[key]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return nil
        }
        guard let raw = cachedPredict(
            infinitive: key,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        ) else { return nil }

        let form = Self.alternativeForm(raw)
        let pronoun = pronounString(person: person, mode: mode, form: form, infinitive: infinitive)
        return pronoun + form
    }

    /// Get all participle forms for a verb in a given voice.
    ///
    /// Returns the **primary** form for each tense.  Use
    /// ``participleAlternative(_:voice:tense:)`` for individual
    /// alternative lookups.
    ///
    ///     conjugator.participles("partir", voice: .activeEtre)
    ///     // -> [.present: "partant", .passeMasculinSingulier: "parti",
    ///     //    .passeFemininSingulier: "partie", ...]
    ///
    /// - Returns: A dictionary mapping each available tense to its participle form.
    public func participles(
        _ infinitive: String,
        voice: Voice,
        homonymIndex: Int? = nil
    ) -> [Tense: String] {
        lock.lock()
        defer { lock.unlock() }
        let key = resolveKey(infinitive, homonymIndex: homonymIndex)
        guard let tenseMap = engine.verbStructure[key]?[voice.rawValue]?["participe"] else {
            return [:]
        }
        var result = [Tense: String]()
        for (tenseKey, personKeys) in tenseMap {
            guard let tense = Tense(rawValue: tenseKey),
                  personKeys.contains("-") else { continue }
            if let form = cachedPredict(
                infinitive: key,
                voice: voice.rawValue,
                mode: "participe",
                tense: tenseKey,
                person: "-"
            ) {
                result[tense] = Self.primaryForm(form)
            }
        }
        return result
    }

    /// Async variant of ``participle(_:voice:tense:)``.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func participle(
        _ infinitive: String,
        voice: Voice,
        tense: Tense,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.participle(infinitive, voice: voice, tense: tense,
                                             homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    /// Async variant of ``participleAlternative(_:voice:tense:)``.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func participleAlternative(
        _ infinitive: String,
        voice: Voice,
        tense: Tense,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.participleAlternative(infinitive, voice: voice, tense: tense,
                                                       homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    /// Async variant of ``getPronoun(_:voice:mode:tense:person:)``.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func getPronoun(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.getPronoun(infinitive, voice: voice, mode: mode,
                                             tense: tense, person: person,
                                             homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    /// Async variant of ``conjugateWithPronoun(_:voice:mode:tense:person:)``.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func conjugateWithPronoun(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugateWithPronoun(infinitive, voice: voice, mode: mode,
                                                      tense: tense, person: person,
                                                      homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    /// Async variant of ``conjugateAlternativeWithPronoun(_:voice:mode:tense:person:)``.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func conjugateAlternativeWithPronoun(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugateAlternativeWithPronoun(infinitive, voice: voice, mode: mode,
                                                                  tense: tense, person: person,
                                                                  homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    // MARK: - Async Initialization

    /// Asynchronously load the conjugation model from a directory path.
    ///
    /// - Parameters:
    ///   - path: Absolute path to the model directory.
    ///   - cacheSize: Maximum number of verbs to cache (default: ``defaultCacheSize``).
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public static func load(modelDirectory path: String, cacheSize: Int = defaultCacheSize) async throws -> Conjugator {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let conjugator = try Conjugator(modelDirectory: path, cacheSize: cacheSize)
                    continuation.resume(returning: conjugator)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Asynchronously load the conjugation model from a directory URL.
    ///
    /// - Parameters:
    ///   - url: File URL to the model directory.
    ///   - cacheSize: Maximum number of verbs to cache (default: ``defaultCacheSize``).
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public static func load(modelDirectory url: URL, cacheSize: Int = defaultCacheSize) async throws -> Conjugator {
        try await load(modelDirectory: url.path, cacheSize: cacheSize)
    }

    /// Asynchronously load the conjugation model from bundled resources.
    ///
    /// - Parameter cacheSize: Maximum number of verbs to cache (default: ``defaultCacheSize``).
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public static func load(cacheSize: Int = defaultCacheSize) async throws -> Conjugator {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let conjugator = try Conjugator(cacheSize: cacheSize)
                    continuation.resume(returning: conjugator)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    // MARK: - Async Conjugation

    /// Conjugate a single form asynchronously.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugate(infinitive, voice: voice, mode: mode,
                                            tense: tense, person: person,
                                            homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    /// Conjugate all persons for a given voice, mode and tense asynchronously.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        homonymIndex: Int? = nil
    ) async -> [Person: String] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugate(infinitive, voice: voice, mode: mode, tense: tense,
                                            homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }

    /// Return the alternative spelling for a single form asynchronously.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func conjugateAlternative(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person,
        homonymIndex: Int? = nil
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugateAlternative(infinitive, voice: voice, mode: mode,
                                                       tense: tense, person: person,
                                                       homonymIndex: homonymIndex)
                continuation.resume(returning: result)
            }
        }
    }
}
