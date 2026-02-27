/**
 * FRConjugation.h — Objective-C wrapper for the French conjugation C library.
 */

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface FRConjugation : NSObject

/**
 * Initialize with the directory containing the exported model files.
 *
 * The directory must contain:
 *   conjugation_encoder.pt, conjugation_bridge.pt,
 *   conjugation_attention.pt, conjugation_decoder.pt,
 *   conjugation_meta.json
 *
 * @param modelDirectory  Path to the directory.
 * @return An initialized instance, or nil if loading failed.
 */
- (nullable instancetype)initWithModelDirectory:(NSString *)modelDirectory;

- (instancetype)init NS_UNAVAILABLE;

/** Number of known verbs. */
@property (nonatomic, readonly) NSInteger verbCount;

/** Check whether a verb is known. */
- (BOOL)hasVerb:(NSString *)infinitive;

/** Check for aspirate h. */
- (BOOL)isHAspire:(NSString *)infinitive;

/** Auxiliary verb(s) — e.g. @[@"avoir"], @[@"être"], @[@"avoir", @"pronominal"]. */
- (NSArray<NSString *> *)auxiliaryForVerb:(NSString *)infinitive;

/**
 * Conjugate a single form.
 *
 * @param infinitive  Verb infinitive (e.g. @"parler").
 * @param mode        Grammatical mode (e.g. @"indicatif").
 * @param tense       Tense (e.g. @"present", @"passe_compose").
 * @param person      Person (e.g. @"1s", @"je").
 * @return The conjugated form, or nil if unavailable.
 */
- (nullable NSString *)conjugate:(NSString *)infinitive
                            mode:(NSString *)mode
                           tense:(NSString *)tense
                          person:(NSString *)person;

/**
 * Get a participle form.
 *
 * @param infinitive  Verb infinitive.
 * @param forme       "present", "passe_sm", "passe_sf", "passe_pm", "passe_pf".
 * @return The participle, or nil if unavailable.
 */
- (nullable NSString *)participle:(NSString *)infinitive
                            forme:(NSString *)forme;

@end

NS_ASSUME_NONNULL_END
