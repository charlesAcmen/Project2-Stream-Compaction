/**
 * @file      main_parameterized.cpp
 * @brief     Parameterized benchmark for scan, compaction, and radix sort.
 *            Outputs CSV files for later analysis and chart generation.
 *
 * Usage:
 *   benchmark.exe --mode scan  --sizes 256,1024,4096,... --output results/
 *   benchmark.exe --mode radix --sizes 256,1024,4096,... --output results/
 *   benchmark.exe --mode all   --sizes 256,1024,4096,... --output results/
 *
 * Each run creates a timestamped subfolder under --output so experiments
 * never overwrite each other.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radixsort.h>
#include <stream_compaction/sharedMem.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(p) _mkdir(p)
#else
#define MKDIR(p) mkdir(p, 0755)
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string timestamp() {
    time_t now = time(nullptr);
    struct tm t;
#ifdef _WIN32
    localtime_s(&t, &now);
#else
    localtime_r(&now, &t);
#endif
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d",
             t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
             t.tm_hour, t.tm_min, t.tm_sec);
    return buf;
}

static bool mkdirRecursive(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find_first_of("/\\", pos + 1)) != std::string::npos) {
        std::string sub = path.substr(0, pos);
        MKDIR(sub.c_str());
    }
    MKDIR(path.c_str());
    return true;
}

static std::vector<int> parseSizes(const char* str) {
    std::vector<int> sizes;
    std::string s(str);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) sizes.push_back(std::stoi(item));
    }
    return sizes;
}

static void genRandomArray(int n, int *a, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        a[i] = rand() & 0x7FFFFFFF; // non-negative
    }
}

static void printUsage(const char* prog) {
    printf("Usage: %s --mode <scan|compact|radix|all> --sizes <n1,n2,...> [--output <dir>]\n", prog);
    printf("\n");
    printf("  --mode    scan      : CPU / Naive / Efficient / Thrust scan comparison\n");
    printf("            compact   : CPU without-scan / with-scan / GPU efficient compact\n");
    printf("            radix     : CPU std::sort vs GPU radix sort comparison\n");
    printf("            sharedmem : Shared-memory vs global-memory scan (n <= 128)\n");
    printf("            all       : run all modes\n");
    printf("  --sizes   Comma-separated array sizes, e.g. 256,1024,4096,16384,...\n");
    printf("  --output  Parent directory for results (default: results/)\n");
    printf("            A timestamped subfolder is created automatically.\n");
}

// ---------------------------------------------------------------------------
// CSV writer (tiny RAII wrapper)
// ---------------------------------------------------------------------------
struct CsvWriter {
    FILE* f = nullptr;
    void open(const std::string& path) { f = fopen(path.c_str(), "w"); }
    ~CsvWriter() { if (f) fclose(f); }
    void header(const char* h) { if (f) fprintf(f, "%s\n", h); }
    void row(const char* fmt, ...) {
        if (!f) return;
        va_list args;
        va_start(args, fmt);
        vfprintf(f, fmt, args);
        va_end(args);
        fprintf(f, "\n");
    }
};

// ---------------------------------------------------------------------------
// Benchmark: Scan
// ---------------------------------------------------------------------------
void runScanBench(const std::vector<int>& sizes, const std::string& outDir) {
    std::string path = outDir + "/scan_comparison.csv";
    CsvWriter csv;
    csv.open(path);
    csv.header("size,cpu_ms,naive_ms,efficient_ms,thrust_ms");

    printf("\n==== SCAN BENCHMARK ====\n");

    for (int n : sizes) {
        printf("  n = %d ... ", n);
        fflush(stdout);

        int *idata = new int[n];
        int *odata = new int[n];
        genRandomArray(n, idata, 42);

        // --- CPU scan ---
        StreamCompaction::CPU::scan(n, odata, idata);
        float cpuMs = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

        // --- Naive ---
        StreamCompaction::Naive::scan(n, odata, idata);
        float naiveMs = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

        // --- Efficient ---
        StreamCompaction::Efficient::scan(n, odata, idata);
        float efficientMs = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

        // --- Thrust ---
        StreamCompaction::Thrust::scan(n, odata, idata);
        float thrustMs = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

        csv.row("%d,%.6f,%.6f,%.6f,%.6f", n, cpuMs, naiveMs, efficientMs, thrustMs);
        printf("cpu=%.3f naive=%.3f eff=%.3f thrust=%.3f ms\n",
               cpuMs, naiveMs, efficientMs, thrustMs);

        delete[] idata;
        delete[] odata;
    }
    printf("  -> wrote %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// Benchmark: Stream Compaction
// ---------------------------------------------------------------------------
void runCompactBench(const std::vector<int>& sizes, const std::string& outDir) {
    std::string path = outDir + "/compact_comparison.csv";
    CsvWriter csv;
    csv.open(path);
    csv.header("size,cpu_compact_without_scan_ms,cpu_compact_with_scan_ms,gpu_compact_efficient_ms");

    printf("\n==== COMPACTION BENCHMARK ====\n");

    for (int n : sizes) {
        printf("  n = %d ... ", n);
        fflush(stdout);

        int *idata = new int[n];
        int *odata = new int[n];
        // Fill with ~50% zeros to exercise compaction meaningfully
        srand(42);
        for (int i = 0; i < n; i++) {
            idata[i] = (rand() % 2) ? 0 : (rand() % 1000 + 1);
        }

        // --- CPU compact without scan ---
        StreamCompaction::CPU::compactWithoutScan(n, odata, idata);
        float cpuNoScanMs = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

        // --- CPU compact with scan ---
        StreamCompaction::CPU::compactWithScan(n, odata, idata);
        float cpuScanMs = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

        // --- GPU work-efficient compact ---
        StreamCompaction::Efficient::compact(n, odata, idata);
        float gpuMs = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();

        csv.row("%d,%.6f,%.6f,%.6f", n, cpuNoScanMs, cpuScanMs, gpuMs);
        printf("cpu_no_scan=%.3f cpu_scan=%.3f gpu_efficient=%.3f ms\n",
               cpuNoScanMs, cpuScanMs, gpuMs);

        delete[] idata;
        delete[] odata;
    }
    printf("  -> wrote %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// Benchmark: Shared-Memory vs Global-Memory Scan
//
// Measurement strategy:
//   At tiny sizes (n <= 128) kernel time is ~1-5 us — comparable to CUDA API
//   overhead.  To get reliable numbers we:
//     1. Allocate device memory ONCE per size (no per-call malloc/free).
//     2. Run WARMUP launches to prime the CUDA context / GPU clock.
//     3. Time a BATCH of ITERS launches with manual cudaEventRecord, then
//        divide by ITERS for the per-scan average.
//     4. Only kernel-launch regions are bracketed by events; memcpy H2D
//        happens once before the batch and memcpy D2H once after.
// ---------------------------------------------------------------------------
void runSharedMemBench(const std::vector<int>& sizes, const std::string& outDir) {
    const int WARMUP = 5;
    const int ITERS  = 200;

    std::string path = outDir + "/sharedmem_comparison.csv";
    CsvWriter csv;
    csv.open(path);
    csv.header("size,naive_global_ms,naive_shared_ms,"
               "efficient_global_ms,efficient_shared_bc_ms,"
               "efficient_shared_nobc_ms");

    printf("\n==== SHARED-MEMORY SCAN BENCHMARK ====\n");
    printf("  (single-block: n <= 1024, %d warmup + %d timed iterations)\n",
           WARMUP, ITERS);

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);

    float elapsedMs;

    for (int n : sizes) {
        printf("  n = %-4d ... ", n);
        fflush(stdout);

        int *idata = new int[n];
        genRandomArray(n, idata, 42);

        int paddedN = 1 << ilog2ceil(n);   // power-of-2 for eff. scan
        int fullBlocks = (n + 127) / 128;  // grid for global-mem kernels
        int naiveIters = ilog2ceil(n);     // log2(n) kernel launches per scan

        // ---- reused device buffers ----
        int *dev_data = nullptr, *dev_buf1 = nullptr, *dev_buf2 = nullptr;
        cudaMalloc(&dev_data, paddedN * sizeof(int));   // efficient scan
        cudaMalloc(&dev_buf1, n * sizeof(int));         // naive scan buf 1
        cudaMalloc(&dev_buf2, n * sizeof(int));         // naive scan buf 2

        // ---------------------------------------------------------------
        // 1. Naive global-memory scan
        // ---------------------------------------------------------------
        {
            cudaMemcpy(dev_buf1, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);

            // One "scan" = naiveIters kernel launches (ping-pong)
            // Warmup
            for (int w = 0; w < WARMUP; w++) {
                for (int d = 1; d <= naiveIters; d++) {
                    int offset = 1 << (d - 1);
                    if (d % 2 == 1)
                        StreamCompaction::Naive::kernNaiveScan<<<fullBlocks, 128>>>(
                            n, offset, dev_buf2, dev_buf1);
                    else
                        StreamCompaction::Naive::kernNaiveScan<<<fullBlocks, 128>>>(
                            n, offset, dev_buf1, dev_buf2);
                }
            }
            cudaDeviceSynchronize();

            // Timed batch
            cudaEventRecord(evStart);
            for (int i = 0; i < ITERS; i++) {
                for (int d = 1; d <= naiveIters; d++) {
                    int offset = 1 << (d - 1);
                    if (d % 2 == 1)
                        StreamCompaction::Naive::kernNaiveScan<<<fullBlocks, 128>>>(
                            n, offset, dev_buf2, dev_buf1);
                    else
                        StreamCompaction::Naive::kernNaiveScan<<<fullBlocks, 128>>>(
                            n, offset, dev_buf1, dev_buf2);
                }
            }
            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);
            cudaEventElapsedTime(&elapsedMs, evStart, evStop);
            float naiveGlobalMs = elapsedMs / ITERS;

            // ---------------------------------------------------------------
            // 2. Naive shared-memory scan
            // ---------------------------------------------------------------
            float naiveSharedMs = 0.f;

            // Prepare: always start from dev_buf1 (contains original data)
            cudaMemcpy(dev_buf2, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);  // ensure clean input

            // Warmup
            size_t naiveSmem = 2 * n * sizeof(int);
            for (int w = 0; w < WARMUP; w++) {
                StreamCompaction::SharedMem::kernNaiveSharedScan<<<1, n, naiveSmem>>>(
                    n, dev_buf1, dev_buf2);
            }
            cudaDeviceSynchronize();

            // Timed batch
            cudaEventRecord(evStart);
            for (int i = 0; i < ITERS; i++) {
                StreamCompaction::SharedMem::kernNaiveSharedScan<<<1, n, naiveSmem>>>(
                    n, dev_buf1, dev_buf2);
            }
            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);
            cudaEventElapsedTime(&elapsedMs, evStart, evStop);
            naiveSharedMs = elapsedMs / ITERS;

            // ---------------------------------------------------------------
            // 3. Work-efficient global-memory scan
            // ---------------------------------------------------------------
            // Zero-pad dev_data and copy input
            cudaMemcpy(dev_data, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);
            if (paddedN > n)
                cudaMemset(dev_data + n, 0, (paddedN - n) * sizeof(int));

            int levels = ilog2ceil(paddedN);

            // Warmup (with initial data setup)
            cudaMemcpy(dev_data, idata, n * sizeof(int),
                       cudaMemcpyHostToDevice);
            if (paddedN > n)
                cudaMemset(dev_data + n, 0, (paddedN - n) * sizeof(int));
            for (int w = 0; w < WARMUP; w++) {
                for (int d = 0; d < levels; d++) {
                    int nThreads = paddedN / (1 << (d + 1));
                    int nBlocks = (nThreads + 127) / 128;
                    StreamCompaction::Efficient::kernUpSweep<<<nBlocks, 128>>>(
                        paddedN, d, dev_data);
                }
                cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
                for (int d = levels - 1; d >= 0; d--) {
                    int nThreads = paddedN / (1 << (d + 1));
                    int nBlocks = (nThreads + 127) / 128;
                    StreamCompaction::Efficient::kernDownSweep<<<nBlocks, 128>>>(
                        paddedN, d, dev_data);
                }
            }
            cudaDeviceSynchronize();

            // Timed batch — no cudaMemcpy/cudaMemset inside:
            // scan work is data-independent so repeating on the same
            // (already-scanned) data measures pure kernel time.
            cudaEventRecord(evStart);
            for (int i = 0; i < ITERS; i++) {
                for (int d = 0; d < levels; d++) {
                    int nThreads = paddedN / (1 << (d + 1));
                    int nBlocks = (nThreads + 127) / 128;
                    StreamCompaction::Efficient::kernUpSweep<<<nBlocks, 128>>>(
                        paddedN, d, dev_data);
                }
                cudaMemset(dev_data + paddedN - 1, 0, sizeof(int));
                for (int d = levels - 1; d >= 0; d--) {
                    int nThreads = paddedN / (1 << (d + 1));
                    int nBlocks = (nThreads + 127) / 128;
                    StreamCompaction::Efficient::kernDownSweep<<<nBlocks, 128>>>(
                        paddedN, d, dev_data);
                }
            }
            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);
            cudaEventElapsedTime(&elapsedMs, evStart, evStop);
            float effGlobalMs = elapsedMs / ITERS;

            // ---------------------------------------------------------------
            // 4. Work-efficient shared-memory scan WITH bank-conflict avoid.
            // ---------------------------------------------------------------
            int smemSlotsBc = paddedN +
                (paddedN - 1) / 32 + 1;       // padded shared-mem slots
            size_t effSmemBc = smemSlotsBc * sizeof(int);

            // Warmup
            for (int w = 0; w < WARMUP; w++) {
                StreamCompaction::SharedMem::kernWorkEfficientSharedScan<<<
                    1, paddedN, effSmemBc>>>(n, paddedN, dev_buf1, dev_buf2);
            }
            cudaDeviceSynchronize();

            // Timed batch
            cudaEventRecord(evStart);
            for (int i = 0; i < ITERS; i++) {
                StreamCompaction::SharedMem::kernWorkEfficientSharedScan<<<
                    1, paddedN, effSmemBc>>>(n, paddedN, dev_buf1, dev_buf2);
            }
            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);
            cudaEventElapsedTime(&elapsedMs, evStart, evStop);
            float effSharedBcMs = elapsedMs / ITERS;

            // ---------------------------------------------------------------
            // 5. Work-efficient shared-memory WITHOUT bank-conflict avoid.
            // ---------------------------------------------------------------
            size_t effSmemNoBc = paddedN * sizeof(int);

            // Warmup
            for (int w = 0; w < WARMUP; w++) {
                StreamCompaction::SharedMem::kernWorkEfficientSharedScanNoBC<<<
                    1, paddedN, effSmemNoBc>>>(n, paddedN, dev_buf1, dev_buf2);
            }
            cudaDeviceSynchronize();

            // Timed batch
            cudaEventRecord(evStart);
            for (int i = 0; i < ITERS; i++) {
                StreamCompaction::SharedMem::kernWorkEfficientSharedScanNoBC<<<
                    1, paddedN, effSmemNoBc>>>(n, paddedN, dev_buf1, dev_buf2);
            }
            cudaEventRecord(evStop);
            cudaEventSynchronize(evStop);
            cudaEventElapsedTime(&elapsedMs, evStart, evStop);
            float effSharedNoBcMs = elapsedMs / ITERS;

            // ---- Write CSV row ----
            csv.row("%d,%.6f,%.6f,%.6f,%.6f,%.6f",
                    n, naiveGlobalMs, naiveSharedMs,
                    effGlobalMs, effSharedBcMs, effSharedNoBcMs);
            printf("nv_gl=%.4f nv_sh=%.4f | ef_gl=%.4f ef_bc=%.4f ef_nbc=%.4f ms\n",
                   naiveGlobalMs, naiveSharedMs,
                   effGlobalMs, effSharedBcMs, effSharedNoBcMs);
        }

        // ---- Cleanup ----
        cudaFree(dev_data);
        cudaFree(dev_buf1);
        cudaFree(dev_buf2);
        delete[] idata;
    }

    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    printf("  -> wrote %s\n", path.c_str());
}

// ---------------------------------------------------------------------------
// Benchmark: Radix Sort
// ---------------------------------------------------------------------------
void runRadixBench(const std::vector<int>& sizes, const std::string& outDir) {
    std::string path = outDir + "/radix_comparison.csv";
    CsvWriter csv;
    csv.open(path);
    csv.header("size,cpu_sort_ms,gpu_radix_ms");

    printf("\n==== RADIX SORT BENCHMARK ====\n");

    for (int n : sizes) {
        printf("  n = %d ... ", n);
        fflush(stdout);

        int *gpuData = new int[n];
        int *cpuData = new int[n];
        genRandomArray(n, gpuData, 2025);
        memcpy(cpuData, gpuData, n * sizeof(int));

        // --- GPU radix sort ---
        StreamCompaction::RadixSort::radixsort(n, gpuData);
        float gpuMs = StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation();

        // --- CPU std::sort ---
        StreamCompaction::RadixSort::timer().startCpuTimer();
        std::sort(cpuData, cpuData + n);
        StreamCompaction::RadixSort::timer().endCpuTimer();
        float cpuMs = StreamCompaction::RadixSort::timer().getCpuElapsedTimeForPreviousOperation();

        csv.row("%d,%.6f,%.6f", n, cpuMs, gpuMs);
        printf("cpu_sort=%.3f gpu_radix=%.3f ms\n", cpuMs, gpuMs);

        delete[] gpuData;
        delete[] cpuData;
    }
    printf("  -> wrote %s\n", path.c_str());
}

// =======================================================================
int main(int argc, char* argv[]) {
    std::string mode;
    std::string sizesStr;
    std::string outputParent = "results";

    // ---- Parse arguments ----
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizesStr = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            outputParent = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (mode.empty() || sizesStr.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    if (mode != "scan" && mode != "compact" && mode != "radix" &&
        mode != "sharedmem" && mode != "all") {
        printf("Error: --mode must be 'scan', 'compact', 'radix', "
               "'sharedmem', or 'all'\n");
        return 1;
    }

    auto sizes = parseSizes(sizesStr.c_str());
    if (sizes.empty()) {
        printf("Error: --sizes must be a comma-separated list, e.g. 256,1024,4096\n");
        return 1;
    }

    // ---- Create output directory ----
    // If --output points to an existing dir, use it directly (no extra timestamp).
    // Otherwise create a timestamped subfolder inside --output.
    std::string outDir;
    {
        struct stat st;
        if (stat(outputParent.c_str(), &st) == 0 && (st.st_mode & S_IFDIR)) {
            // Already exists (e.g. created by driver script) -- use as-is
            outDir = outputParent;
        } else {
            outDir = outputParent + "/" + timestamp();
        }
    }
    mkdirRecursive(outDir);
    printf("Output directory: %s\n", outDir.c_str());

    // ---- Run benchmarks ----
    if (mode == "scan" || mode == "all") {
        runScanBench(sizes, outDir);
    }
    if (mode == "compact" || mode == "all") {
        runCompactBench(sizes, outDir);
    }
    if (mode == "radix" || mode == "all") {
        runRadixBench(sizes, outDir);
    }
    if (mode == "sharedmem" || mode == "all") {
        // For shared-memory mode, clamp sizes to blockSize (1024)
        std::vector<int> clampedSizes;
        for (int s : sizes) {
            if (s <= 1024) {
                clampedSizes.push_back(s);
            } else {
                printf("  NOTE: skipping n=%d (> blockSize=1024 for "
                       "shared-mem scan)\n", s);
            }
        }
        if (!clampedSizes.empty()) {
            runSharedMemBench(clampedSizes, outDir);
        } else {
            printf("\n  SKIP sharedmem: all sizes > blockSize=1024\n");
        }
    }

    printf("\nDone.\n");
    return 0;
}
