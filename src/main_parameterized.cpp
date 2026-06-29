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
    printf("Usage: %s --mode <scan|radix|all> --sizes <n1,n2,...> [--output <dir>]\n", prog);
    printf("\n");
    printf("  --mode    scan  : CPU / Naive / Efficient / Thrust scan comparison\n");
    printf("            radix : CPU std::sort vs GPU radix sort comparison\n");
    printf("            all   : run both modes\n");
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

    if (mode != "scan" && mode != "radix" && mode != "all") {
        printf("Error: --mode must be 'scan', 'radix', or 'all'\n");
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
    if (mode == "radix" || mode == "all") {
        runRadixBench(sizes, outDir);
    }

    printf("\nDone.\n");
    return 0;
}
