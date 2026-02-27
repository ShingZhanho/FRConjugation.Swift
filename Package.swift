// swift-tools-version:5.6
import PackageDescription

let package = Package(
    name: "FRConjugation",
    platforms: [
        .macOS(.v10_15),
        .iOS(.v13)
    ],
    products: [
        .library(
            name: "FRConjugation",
            targets: ["FRConjugation"]
        )
    ],
    targets: [
        .target(
            name: "FRConjugation",
            path: "swift_lib/Sources/FRConjugation",
            resources: [
                .copy("Resources")
            ]
        ),
        .target(
            name: "CFRConjugation",
            path: "swift_lib/Sources/CFRConjugation"
        ),
        .testTarget(
            name: "FRConjugationTests",
            path: "swift_lib/Tests/FRConjugationTests"
        )
    ]
)
