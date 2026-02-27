// Tensor.swift — Lightweight dense tensor backed by a flat Float array.
//
// Uses Apple's Accelerate framework (vDSP / cblas) for matrix operations.

import Accelerate
import Foundation

/// A simple dense tensor storing row-major Float data.
///
/// This is an internal type used by the inference engine — it is *not*
/// a general-purpose tensor library.  Only the operations needed for
/// seq2seq inference are implemented.
struct Tensor {
    /// The flat, row-major data.
    var data: [Float]
    /// The tensor shape (e.g. [rows, cols] for a matrix).
    let shape: [Int]

    /// Total number of elements.
    var count: Int { data.count }

    // MARK: - Initializers

    /// Create a tensor filled with zeros.
    static func zeros(_ shape: [Int]) -> Tensor {
        let n = shape.reduce(1, *)
        return Tensor(data: [Float](repeating: 0, count: n), shape: shape)
    }

    /// Create a 1-D tensor from an array of floats.
    static func vector(_ v: [Float]) -> Tensor {
        Tensor(data: v, shape: [v.count])
    }

    /// Create a tensor by loading `count` Float32 values from a pointer.
    init(from pointer: UnsafeRawPointer, count: Int, shape: [Int]) {
        let buf = pointer.bindMemory(to: Float.self, capacity: count)
        self.data = Array(UnsafeBufferPointer(start: buf, count: count))
        self.shape = shape
    }
}

// MARK: - Element Access

extension Tensor {
    /// 1-D subscript.
    subscript(i: Int) -> Float {
        get { data[i] }
        set { data[i] = newValue }
    }

    /// 2-D subscript (row, col) — assumes shape has 2 dimensions.
    subscript(r: Int, c: Int) -> Float {
        get { data[r * shape[1] + c] }
        set { data[r * shape[1] + c] = newValue }
    }

    /// Return a row slice from a 2-D tensor as a 1-D tensor.
    func row(_ r: Int) -> Tensor {
        let cols = shape[1]
        let start = r * cols
        return Tensor(data: Array(data[start..<start + cols]), shape: [cols])
    }
}

// MARK: - Arithmetic (vDSP-accelerated)

extension Tensor {
    /// Element-wise addition: self + other (same shape).
    func add(_ other: Tensor) -> Tensor {
        assert(count == other.count)
        var result = [Float](repeating: 0, count: count)
        vDSP_vadd(data, 1, other.data, 1, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }

    /// Element-wise addition of a scalar.
    func add(_ scalar: Float) -> Tensor {
        var s = scalar
        var result = [Float](repeating: 0, count: count)
        vDSP_vsadd(data, 1, &s, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }

    /// Element-wise multiplication: self * other (same shape).
    func mul(_ other: Tensor) -> Tensor {
        assert(count == other.count)
        var result = [Float](repeating: 0, count: count)
        vDSP_vmul(data, 1, other.data, 1, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }

    /// Scalar multiplication.
    func mul(_ scalar: Float) -> Tensor {
        var s = scalar
        var result = [Float](repeating: 0, count: count)
        vDSP_vsmul(data, 1, &s, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }

    /// Element-wise subtraction: self - other.
    func sub(_ other: Tensor) -> Tensor {
        // vDSP_vsub computes B - A, so swap args.
        assert(count == other.count)
        var result = [Float](repeating: 0, count: count)
        vDSP_vsub(other.data, 1, data, 1, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }

    /// Negate all elements: -self.
    func neg() -> Tensor {
        var result = [Float](repeating: 0, count: count)
        vDSP_vneg(data, 1, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }
}

// MARK: - Matrix Operations (cblas / vDSP)

extension Tensor {
    /// Matrix-vector product: self (M×K) * vec (K) → result (M).
    func matvec(_ vec: Tensor) -> Tensor {
        let M = shape[0]
        let K = shape[1]
        assert(vec.count == K, "matvec: incompatible shapes \(shape) and [\(vec.count)]")
        var result = [Float](repeating: 0, count: M)
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            Int32(M), Int32(K),
            1.0, data, Int32(K),
            vec.data, 1,
            0.0, &result, 1
        )
        return Tensor(data: result, shape: [M])
    }

    /// Matrix multiply: self (M×K) × other (K×N) → result (M×N).
    func matmul(_ other: Tensor) -> Tensor {
        let M = shape[0]
        let K = shape[1]
        let N = other.shape[1]
        assert(other.shape[0] == K)
        var result = [Float](repeating: 0, count: M * N)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(M), Int32(N), Int32(K),
            1.0, data, Int32(K),
            other.data, Int32(N),
            0.0, &result, Int32(N)
        )
        return Tensor(data: result, shape: [M, N])
    }
}

// MARK: - Activation Functions

extension Tensor {
    /// Element-wise tanh.
    func tanh() -> Tensor {
        var n = Int32(count)
        var result = [Float](repeating: 0, count: count)
        vvtanhf(&result, data, &n)
        return Tensor(data: result, shape: shape)
    }

    /// Element-wise sigmoid: 1 / (1 + exp(-x)).
    func sigmoid() -> Tensor {
        // Compute -x, then exp(-x), then 1/(1+exp(-x))
        var n = Int32(count)
        var negated = [Float](repeating: 0, count: count)
        vDSP_vneg(data, 1, &negated, 1, vDSP_Length(count))
        var exped = [Float](repeating: 0, count: count)
        vvexpf(&exped, negated, &n)
        // exped = 1 + exp(-x)
        var one: Float = 1.0
        vDSP_vsadd(exped, 1, &one, &exped, 1, vDSP_Length(count))
        // result = 1 / (1 + exp(-x))
        var result = [Float](repeating: 0, count: count)
        vvrecf(&result, exped, &n)
        return Tensor(data: result, shape: shape)
    }

    /// Softmax over the entire flat array (used for 1-D score vectors).
    func softmax() -> Tensor {
        // For numerical stability: subtract max first.
        var maxVal: Float = 0
        vDSP_maxv(data, 1, &maxVal, vDSP_Length(count))
        var negMax = -maxVal
        var shifted = [Float](repeating: 0, count: count)
        vDSP_vsadd(data, 1, &negMax, &shifted, 1, vDSP_Length(count))
        var n = Int32(count)
        var exped = [Float](repeating: 0, count: count)
        vvexpf(&exped, shifted, &n)
        var sum: Float = 0
        vDSP_sve(exped, 1, &sum, vDSP_Length(count))
        var result = [Float](repeating: 0, count: count)
        var invSum = 1.0 / sum
        vDSP_vsmul(exped, 1, &invSum, &result, 1, vDSP_Length(count))
        return Tensor(data: result, shape: shape)
    }

    /// Index of the maximum element.
    func argmax() -> Int {
        var maxVal: Float = 0
        var maxIdx: vDSP_Length = 0
        vDSP_maxvi(data, 1, &maxVal, &maxIdx, vDSP_Length(count))
        return Int(maxIdx)
    }
}

// MARK: - Concatenation

extension Tensor {
    /// Concatenate two 1-D tensors.
    static func cat(_ a: Tensor, _ b: Tensor) -> Tensor {
        Tensor(data: a.data + b.data, shape: [a.count + b.count])
    }

    /// Concatenate multiple 1-D tensors.
    static func cat(_ tensors: [Tensor]) -> Tensor {
        let totalCount = tensors.reduce(0) { $0 + $1.count }
        var result = [Float]()
        result.reserveCapacity(totalCount)
        for t in tensors {
            result.append(contentsOf: t.data)
        }
        return Tensor(data: result, shape: [totalCount])
    }
}
