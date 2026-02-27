// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FRConjugation",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "FRConjugation", targets: ["FRConjugation"]),
    ],
    targets: [
        // C bridge — exposes the C API header from c_wrapper.
        // Link against libfrconjugation (built via CMake from c_wrapper/).
        .target(
            name: "CFRConjugation",
            path: "Sources/CFRConjugation",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedLibrary("frconjugation"),
            ]
        ),
        // Pure-Swift idiomatic API built on top of the C bridge.
        .target(
            name: "FRConjugation",
            dependencies: ["CFRConjugation"]
        ),
        .testTarget(
            name: "FRConjugationTests",
            dependencies: ["FRConjugation"]
        ),
    ]
)
