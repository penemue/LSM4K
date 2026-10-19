package io.github.martinhaeusler.lsm4k.compressor.benchmark

import io.github.martinhaeusler.lsm4k.compressor.api.Compressor
import io.github.martinhaeusler.lsm4k.compressor.api.Compressors
import io.github.martinhaeusler.lsm4k.model.command.Command
import io.github.martinhaeusler.lsm4k.util.TSN
import io.github.martinhaeusler.lsm4k.util.bytes.Bytes
import org.junit.jupiter.api.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * Lightweight comparison of the registered [Compressor] implementations
 * (`gzip`, `snappy`, `lz4`, `zstd`) along the three dimensions that matter
 * for an LSM storage engine:
 *
 *  - **compression ratio** (`compressed / original`, smaller is better),
 *  - **compression throughput** (MiB/s),
 *  - **decompression throughput** (MiB/s) - the most important one, since
 *    blocks are read far more often than they are written.
 *
 * This is intentionally **not** a JMH benchmark: it requires no build changes
 * and is fully self-contained. Absolute numbers depend on the hardware and JVM,
 * so focus on the *relative* differences between the algorithms.
 *
 * Each algorithm is measured against several data profiles, because the input
 * shape dominates the result: random data is essentially incompressible, while
 * realistic LSM key/value data compresses well.
 */
class CompressorBenchmarkTest {

    companion object {

        /**
         * The compressors under comparison, by their [Compressor.uniqueName].
         */
        private val COMPRESSOR_NAMES = listOf("gzip", "snappy", "lz4", "zstd")

        /**
         * Size of every generated dataset. Matches the engine's default `maxBlockSize`
         * ([io.github.martinhaeusler.lsm4k.api.LSM4KConfiguration] -> 8 MiB), since the
         * compressor always operates on a single block at a time.
         */
        private const val PAYLOAD_SIZE_BYTES = 8 * 1024 * 1024 // 8 MiB

        /**
         * Untimed iterations to let the JIT warm up before measuring.
         */
        private const val WARMUP_ITERATIONS = 3

        /**
         * Timed iterations; the reported time is the average over these runs.
         */
        private const val MEASUREMENT_ITERATIONS = 8

        /**
         * Fixed seed so the generated data (and thus the benchmark) is reproducible.
         */
        private const val RANDOM_SEED = 42L

        /**
         * Results are also written here, because Gradle does not forward test stdout to the console.
         */
        private const val RESULTS_FILE = "build/compressor-benchmark-results.txt"

        private const val BYTES_PER_MIB = 1024.0 * 1024.0
        private const val NANOS_PER_SECOND = 1_000_000_000.0

        // Layout of the synthetic binary record used by the realistic dataset (mirrors a serialized DB row).
        private const val KEY_SIZE_BYTES = 16
        private const val KEY_TABLE_PREFIX = 7
        private const val VALUE_SIZE_BYTES = 64
        private const val RECORD_FORMAT_VERSION: Byte = 1
        private const val BASE_TIMESTAMP_MILLIS = 1_700_000_000_000L
        private const val MILLIS_PER_DAY = 86_400_000
        private const val NAME_FIELD_SIZE_BYTES = 24
        private val NAME_VOCABULARY = listOf("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
    }

    @Test
    fun compareCompressors() {
        val datasets = listOf(
            Dataset("highly-compressible", generateHighlyCompressible()),
            Dataset("realistic-lsm-block", generateRealisticLsmData()),
            Dataset("incompressible-random", generateRandom()),
        )
        val compressors = COMPRESSOR_NAMES.map(Compressors::getCompressorForName)

        val report = buildString {
            for (dataset in datasets) {
                appendDatasetHeader(dataset)
                for (compressor in compressors) {
                    appendResultRow(benchmark(compressor, dataset.data))
                }
            }
        }

        println(report)
        File(RESULTS_FILE).apply { parentFile?.mkdirs() }.writeText(report)
    }

    /**
     * Runs the warm-up, verifies round-trip correctness and measures compression
     * and decompression throughput for a single [compressor] on the given [data].
     */
    private fun benchmark(compressor: Compressor, data: ByteArray): BenchmarkResult {
        // Warm up both directions so the JIT has compiled the hot paths.
        repeat(WARMUP_ITERATIONS) {
            val warmCompressed = compressor.compress(data)
            compressor.decompress(warmCompressed)
        }

        val compressed = compressor.compress(data)

        // Sanity check: compression must be lossless, otherwise the numbers are meaningless.
        val roundTrip = compressor.decompress(compressed)
        check(roundTrip.contentEquals(data)) {
            "Round-trip failed for compressor '${compressor.uniqueName}': decompressed data differs from the original!"
        }

        // 'blackhole' prevents the JIT from optimizing the measured work away.
        var blackhole = 0L

        val compressStart = System.nanoTime()
        repeat(MEASUREMENT_ITERATIONS) {
            blackhole += compressor.compress(data).size
        }
        val avgCompressNanos = (System.nanoTime() - compressStart).toDouble() / MEASUREMENT_ITERATIONS

        val decompressStart = System.nanoTime()
        repeat(MEASUREMENT_ITERATIONS) {
            blackhole += compressor.decompress(compressed).size
        }
        val avgDecompressNanos = (System.nanoTime() - decompressStart).toDouble() / MEASUREMENT_ITERATIONS

        check(blackhole > 0) { "Unexpected empty benchmark result." }

        return BenchmarkResult(
            compressorName = compressor.uniqueName,
            originalSize = data.size,
            compressedSize = compressed.size,
            compressThroughputMibPerSec = throughputMibPerSec(data.size, avgCompressNanos),
            decompressThroughputMibPerSec = throughputMibPerSec(data.size, avgDecompressNanos),
        )
    }

    private fun throughputMibPerSec(byteCount: Int, nanos: Double): Double {
        val seconds = nanos / NANOS_PER_SECOND
        val mib = byteCount / BYTES_PER_MIB
        return mib / seconds
    }

    // region DATA GENERATORS

    /**
     * Data with a lot of repetition - represents the best case for any compressor.
     */
    private fun generateHighlyCompressible(): ByteArray {
        val pattern = "The quick brown fox jumps over the lazy dog. ".toByteArray(Charsets.UTF_8)
        return ByteArray(PAYLOAD_SIZE_BYTES) { pattern[it % pattern.size] }
    }

    /**
     * Emulates a serialized LSM data block as written by the engine: a sequence of
     * binary [Command]s (PUT), each encoded via [Command.writeToStream] in the exact
     * on-disk layout - `opcode | keyLength | key | tsn | valueLength | value`.
     *
     * Keys are sorted, prefix-sharing binary primary keys; values are fixed-layout
     * binary records mixing constant headers, clustered timestamps, low-entropy enums
     * and incompressible blobs - i.e. the real shape the engine compresses on disk,
     * not human-readable JSON.
     */
    private fun generateRealisticLsmData(): ByteArray {
        val random = Random(RANDOM_SEED)
        var tsn: TSN = 1L
        var recordId = 0L
        var serializedBytes = 0
        val commands = buildList {
            while (serializedBytes < PAYLOAD_SIZE_BYTES) {
                val command = Command.put(binaryKey(recordId++), tsn++, binaryRecordValue(random))
                add(command)
                serializedBytes += command.byteSize
            }
        }
        // The engine flushes commands into a block in sorted order; reproduce that so the
        // adjacent-key prefix sharing matches what real on-disk blocks actually look like.
        val output = ByteArrayOutputStream(serializedBytes)
        commands.sorted().forEach { it.writeToStream(output) }
        // Truncate to the exact payload size so all datasets stay equally sized and comparable.
        return output.toByteArray().copyOf(PAYLOAD_SIZE_BYTES)
    }

    /**
     * A 16-byte binary primary key: constant table prefix + big-endian (byte-wise sortable) record id.
     */
    private fun binaryKey(recordId: Long): Bytes {
        val buffer = ByteBuffer.allocate(KEY_SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
        buffer.putInt(KEY_TABLE_PREFIX)                                        // table/index id shared by every key -> prefix sharing
        buffer.putInt(0)                                                       // reserved
        buffer.putLong(recordId)                                               // monotonically increasing primary key
        return Bytes.wrap(buffer.array())
    }

    /**
     * A fixed-layout binary record value, the way a serialized DB row would look on disk.
     */
    private fun binaryRecordValue(random: Random): Bytes {
        val buffer = ByteBuffer.allocate(VALUE_SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        buffer.put(RECORD_FORMAT_VERSION)                                      // constant header byte
        buffer.put(random.nextInt(4).toByte())                                 // low-entropy flags/enum
        buffer.putShort(random.nextInt(1000).toShort())                        // small categorical value
        buffer.putInt(random.nextInt(1_000_000))                               // e.g. an amount in cents
        buffer.putLong(BASE_TIMESTAMP_MILLIS + random.nextInt(MILLIS_PER_DAY)) // clustered timestamp (high bytes constant)
        buffer.putLong(random.nextLong())                                      // high-entropy account id
        buffer.put(random.nextBytes(16))                                       // incompressible blob (UUID/hash)
        val name = NAME_VOCABULARY.random(random).toByteArray(Charsets.US_ASCII)
        val nameField = ByteArray(NAME_FIELD_SIZE_BYTES)                       // fixed-width, zero-padded text column
        name.copyInto(nameField, endIndex = minOf(name.size, NAME_FIELD_SIZE_BYTES))
        buffer.put(nameField)
        return Bytes.wrap(buffer.array())
    }

    /**
     * Pseudo-random data - effectively incompressible; shows the per-algorithm overhead.
     */
    private fun generateRandom(): ByteArray {
        return Random(RANDOM_SEED).nextBytes(PAYLOAD_SIZE_BYTES)
    }

    // endregion

    // region OUTPUT

    private fun StringBuilder.appendDatasetHeader(dataset: Dataset) {
        appendLine()
        appendLine("=".repeat(78))
        appendLine("Dataset: ${dataset.name}  (${"%.2f".format(dataset.data.size / BYTES_PER_MIB)} MiB)")
        appendLine("=".repeat(78))
        appendLine("%-8s | %-12s | %-12s | %-16s | %-16s".format("algo", "ratio", "saved", "compress MiB/s", "decompress MiB/s"))
        appendLine("-".repeat(78))
    }

    private fun StringBuilder.appendResultRow(result: BenchmarkResult) {
        val ratio = result.compressedSize.toDouble() / result.originalSize
        val savedPercent = (1.0 - ratio) * 100.0
        appendLine(
            "%-8s | %-12s | %-12s | %-16s | %-16s".format(
                result.compressorName,
                "%.4f".format(ratio),
                "%.2f%%".format(savedPercent),
                "%.1f".format(result.compressThroughputMibPerSec),
                "%.1f".format(result.decompressThroughputMibPerSec),
            )
        )
    }

    // endregion

    // region VALUE TYPES

    private class Dataset(
        val name: String,
        val data: ByteArray,
    )

    private class BenchmarkResult(
        val compressorName: String,
        val originalSize: Int,
        val compressedSize: Int,
        val compressThroughputMibPerSec: Double,
        val decompressThroughputMibPerSec: Double,
    )

    // endregion
}
