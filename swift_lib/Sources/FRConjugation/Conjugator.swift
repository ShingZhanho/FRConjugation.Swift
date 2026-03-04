// Conjugator.swift — Idiomatic Swift interface to the French verb conjugation model.

import Foundation

/// A French verb conjugation engine backed by a character-level seq2seq
/// neural network with Bahdanau attention.
///
/// ## Overview
///
/// `Conjugator` loads a pre-trained model from disk and exposes a fully
/// typed Swift API for conjugating French verbs.  The model covers all
/// five grammatical voices (active-avoir, active-être, active, passive,
/// pronominal) with 12 gender-explicit person keys.
///
/// All conjugation forms — including compound tenses and participles —
/// are predicted directly by the neural model (with an exception table
/// for the tiny fraction it gets wrong).
///
/// ```swift
/// let c = try Conjugator()
///
/// // Single form
/// c.conjugate("aller", voice: .activeEtre, mode: .indicatif,
///             tense: .present, person: .firstSingularMasculine)
/// // → "vais"
///
/// // All persons for a tense
/// let forms = c.conjugate("finir", voice: .activeAvoir,
///                         mode: .indicatif, tense: .imparfait)
/// // → [.firstSingularMasculine: "finissais", ...]
///
/// // Discover valid voices for a verb
/// c.voices("aller")
/// // → [.activeEtre, .pronominal]
/// ```
///
/// ## Caching
///
/// Each `Conjugator` instance maintains an internal **LRU cache** that
/// stores previously-predicted forms indexed by verb infinitive.  The
/// cache size is measured in **verbs** — all forms for the same verb
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
    ///     // First call — sets cache to 128 verbs:
    ///     let fr = Conjugator.getShared(cacheSize: 128)
    ///
    ///     // Later calls — returns the same instance (cacheSize ignored):
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

    /// Reset the shared singleton.  Internal — used by tests to ensure
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
    public var verbCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return engine.knownVerbs.count
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
    public func hasVerb(_ infinitive: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return engine.knownVerbs.contains(infinitive)
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

    // MARK: - Structure Queries

    /// List available voices for a verb.
    ///
    ///     conjugator.voices("aller")
    ///     // → [.activeEtre, .pronominal]
    public func voices(_ infinitive: String) -> [Voice] {
        lock.lock()
        defer { lock.unlock() }
        guard let struct_ = engine.verbStructure[infinitive] else { return [] }
        return struct_.keys.sorted().compactMap { Voice(rawValue: $0) }
    }

    /// List available modes for a verb in a given voice.
    public func modes(_ infinitive: String, voice: Voice) -> [Mode] {
        lock.lock()
        defer { lock.unlock() }
        guard let voiceStruct = engine.verbStructure[infinitive]?[voice.rawValue] else { return [] }
        return voiceStruct.keys.sorted().compactMap { Mode(rawValue: $0) }
    }

    /// List available tenses for a verb in a given voice and mode.
    public func tenses(_ infinitive: String, voice: Voice, mode: Mode) -> [Tense] {
        lock.lock()
        defer { lock.unlock() }
        guard let modeStruct = engine.verbStructure[infinitive]?[voice.rawValue]?[mode.rawValue] else { return [] }
        return modeStruct.keys.sorted().compactMap { Tense(rawValue: $0) }
    }

    /// List available persons for a verb in a given voice, mode and tense.
    public func persons(_ infinitive: String, voice: Voice, mode: Mode, tense: Tense) -> [Person] {
        lock.lock()
        defer { lock.unlock() }
        guard let persons = engine.verbStructure[infinitive]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue] else { return [] }
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

    // MARK: - Conjugation (Single Form)

    /// Conjugate a single form.
    ///
    ///     conjugator.conjugate("aller", voice: .activeEtre,
    ///         mode: .indicatif, tense: .present,
    ///         person: .firstSingularMasculine)
    ///     // → "vais"
    ///
    /// - Returns: The conjugated form, or `nil` if the combination is
    ///   invalid or the verb is unknown.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense,
        person: Person
    ) -> String? {
        lock.lock()
        defer { lock.unlock() }

        // Validate against verb_structure
        guard let persons = engine.verbStructure[infinitive]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue],
              persons.contains(person.rawValue) else {
            return nil
        }

        return cachedPredict(
            infinitive: infinitive,
            voice: voice.rawValue,
            mode: mode.rawValue,
            tense: tense.rawValue,
            person: person.rawValue
        )
    }

    /// Conjugate all persons for a given voice, mode and tense.
    ///
    /// - Returns: A dictionary mapping each valid person to its form.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode,
        tense: Tense
    ) -> [Person: String] {
        lock.lock()
        defer { lock.unlock() }

        guard let personKeys = engine.verbStructure[infinitive]?[voice.rawValue]?[mode.rawValue]?[tense.rawValue] else {
            return [:]
        }

        var result = [Person: String](minimumCapacity: personKeys.count)
        for pKey in personKeys {
            guard let person = Person(rawValue: pKey) else { continue }
            if let form = cachedPredict(
                infinitive: infinitive,
                voice: voice.rawValue,
                mode: mode.rawValue,
                tense: tense.rawValue,
                person: pKey
            ) {
                result[person] = form
            }
        }
        return result
    }

    /// Conjugate all tenses and persons for a voice and mode.
    ///
    /// - Returns: A nested dictionary: tense → person → form.
    public func conjugate(
        _ infinitive: String,
        voice: Voice,
        mode: Mode
    ) -> [Tense: [Person: String]] {
        lock.lock()
        defer { lock.unlock() }

        guard let modeStruct = engine.verbStructure[infinitive]?[voice.rawValue]?[mode.rawValue] else {
            return [:]
        }

        var result = [Tense: [Person: String]]()
        for (tenseKey, personKeys) in modeStruct {
            guard let tense = Tense(rawValue: tenseKey) else { continue }
            var tenseResult = [Person: String](minimumCapacity: personKeys.count)
            for pKey in personKeys {
                guard let person = Person(rawValue: pKey) else { continue }
                if let form = cachedPredict(
                    infinitive: infinitive,
                    voice: voice.rawValue,
                    mode: mode.rawValue,
                    tense: tenseKey,
                    person: pKey
                ) {
                    tenseResult[person] = form
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
    /// - Returns: A nested dictionary: mode → tense → person → form.
    public func conjugate(
        _ infinitive: String,
        voice: Voice
    ) -> [Mode: [Tense: [Person: String]]] {
        lock.lock()
        defer { lock.unlock() }

        guard let voiceStruct = engine.verbStructure[infinitive]?[voice.rawValue] else {
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
                        infinitive: infinitive,
                        voice: voice.rawValue,
                        mode: modeKey,
                        tense: tenseKey,
                        person: pKey
                    ) {
                        tenseResult[person] = form
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
    /// - Returns: A nested dictionary: voice → mode → tense → person → form,
    ///   or `nil` if the verb is unknown.
    public func conjugate(
        _ infinitive: String
    ) -> [Voice: [Mode: [Tense: [Person: String]]]]? {
        lock.lock()
        defer { lock.unlock() }

        guard let verbStruct = engine.verbStructure[infinitive] else {
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
                            infinitive: infinitive,
                            voice: voiceKey,
                            mode: modeKey,
                            tense: tenseKey,
                            person: pKey
                        ) {
                            tenseResult[person] = form
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
    ///     // → "parties"
    ///
    ///     conjugator.participle("parler", voice: .activeAvoir,
    ///                           tense: .present)
    ///     // → "parlant"
    ///
    /// - Parameters:
    ///   - infinitive: The verb infinitive.
    ///   - voice: The grammatical voice.
    ///   - tense: Which participle form to retrieve (e.g. `.present`,
    ///     `.passeMasculinSingulier`, `.passeFemininPluriel`).
    /// - Returns: The participle string, or `nil` if unavailable.
    public func participle(
        _ infinitive: String,
        voice: Voice,
        tense: Tense
    ) -> String? {
        lock.lock()
        defer { lock.unlock() }
        guard let persons = engine.verbStructure[infinitive]?[voice.rawValue]?["participe"]?[tense.rawValue],
              persons.contains("-") else {
            return nil
        }
        return cachedPredict(
            infinitive: infinitive,
            voice: voice.rawValue,
            mode: "participe",
            tense: tense.rawValue,
            person: "-"
        )
    }

    /// Get all participle forms for a verb in a given voice.
    ///
    ///     conjugator.participles("partir", voice: .activeEtre)
    ///     // → [.present: "partant", .passeMasculinSingulier: "parti",
    ///     //    .passeFemininSingulier: "partie", ...]
    ///
    /// - Returns: A dictionary mapping each available tense to its participle form.
    public func participles(
        _ infinitive: String,
        voice: Voice
    ) -> [Tense: String] {
        lock.lock()
        defer { lock.unlock() }
        guard let tenseMap = engine.verbStructure[infinitive]?[voice.rawValue]?["participe"] else {
            return [:]
        }
        var result = [Tense: String]()
        for (tenseKey, personKeys) in tenseMap {
            guard let tense = Tense(rawValue: tenseKey),
                  personKeys.contains("-") else { continue }
            if let form = cachedPredict(
                infinitive: infinitive,
                voice: voice.rawValue,
                mode: "participe",
                tense: tenseKey,
                person: "-"
            ) {
                result[tense] = form
            }
        }
        return result
    }

    /// Async variant of ``participle(_:voice:tense:)``.
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func participle(
        _ infinitive: String,
        voice: Voice,
        tense: Tense
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.participle(infinitive, voice: voice, tense: tense)
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
        person: Person
    ) async -> String? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugate(infinitive, voice: voice, mode: mode,
                                            tense: tense, person: person)
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
        tense: Tense
    ) async -> [Person: String] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = self.conjugate(infinitive, voice: voice, mode: mode, tense: tense)
                continuation.resume(returning: result)
            }
        }
    }
}
