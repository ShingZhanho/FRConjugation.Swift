/**
 * FRConjugation.m — Objective-C wrapper implementation.
 *
 * Bridges the C API (conjugation.h) into idiomatic Objective-C.
 * Link against libfrconjugation when building.
 */

#import "FRConjugation.h"
#import "conjugation.h"

static const NSUInteger kBufferSize = 512;

@implementation FRConjugation {
    FRConjugationModel *_model;
}

#pragma mark - Lifecycle

- (nullable instancetype)initWithModelDirectory:(NSString *)modelDirectory
{
    self = [super init];
    if (self) {
        _model = fr_conjugation_load([modelDirectory fileSystemRepresentation]);
        if (!_model) {
            NSLog(@"FRConjugation: failed to load model from %@", modelDirectory);
            return nil;
        }
    }
    return self;
}

- (void)dealloc
{
    if (_model) {
        fr_conjugation_free(_model);
        _model = NULL;
    }
}

#pragma mark - Properties

- (NSInteger)verbCount
{
    return (NSInteger)fr_conjugation_verb_count(_model);
}

#pragma mark - Query

- (BOOL)hasVerb:(NSString *)infinitive
{
    return fr_conjugation_has_verb(_model, [infinitive UTF8String]);
}

- (BOOL)isHAspire:(NSString *)infinitive
{
    return fr_conjugation_is_h_aspire(_model, [infinitive UTF8String]);
}

- (NSArray<NSString *> *)auxiliaryForVerb:(NSString *)infinitive
{
    char buf[kBufferSize];
    int n = fr_conjugation_auxiliary(_model, [infinitive UTF8String], buf, sizeof(buf));
    if (n <= 0) return @[];
    NSString *raw = [[NSString alloc] initWithUTF8String:buf];
    return [raw componentsSeparatedByString:@","];
}

#pragma mark - Conjugation

- (nullable NSString *)conjugate:(NSString *)infinitive
                            mode:(NSString *)mode
                           tense:(NSString *)tense
                          person:(NSString *)person
{
    char buf[kBufferSize];
    int n = fr_conjugation_conjugate(
        _model,
        [infinitive UTF8String],
        [mode UTF8String],
        [tense UTF8String],
        [person UTF8String],
        buf, sizeof(buf)
    );
    if (n <= 0) return nil;
    return [[NSString alloc] initWithUTF8String:buf];
}

- (nullable NSString *)participle:(NSString *)infinitive
                            forme:(NSString *)forme
{
    char buf[kBufferSize];
    int n = fr_conjugation_get_participle(
        _model,
        [infinitive UTF8String],
        [forme UTF8String],
        buf, sizeof(buf)
    );
    if (n <= 0) return nil;
    return [[NSString alloc] initWithUTF8String:buf];
}

@end
