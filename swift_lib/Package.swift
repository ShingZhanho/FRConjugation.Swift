// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FRConjugation",
    platforms: [.macOS(.v12), .iOS(.v15)],
    products: [
        .library(name: "FRConjugation", targets: ["FRConjugation"]),
    ],
    targets: [
        // Pure-Swift conjugation engine — no C or LibTorch dependency.
        .target(
            name: "FRConjugation",
            resources: [
                .copy("Resources/model.json"),
                .copy("Resources/weights.bin"),
            ]
        ),
        .testTarget(
            name: "FRConjugationTests",
            dependencies: ["FRConjugation"]
        ),
    ]
)
