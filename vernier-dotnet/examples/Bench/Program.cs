// Bench/Program.cs — vernier-dotnet binding benchmark.
//
// Generates a synthetic sinusoidal image and times DetectPeriodic()
// across N iterations.  Prints one line to stdout:
//
//     binding=dotnet backend=cpu size=512x512 iters=N mean=X.XXms best=X.XXms
//
// When --backend cuda is requested but CUDA is unavailable the process exits
// with code 2 and prints a message to stderr; the orchestrator treats this as
// "n/a".
//
// Build and run:
//   cargo build -p vernier-cabi --release
//   LD_LIBRARY_PATH=<repo>/vernier-rs/target/release \
//       dotnet run --project vernier-dotnet/examples/Bench/Bench.csproj \
//       -- [--size 512] [--iters 50] [--backend cpu|cuda]

using System;
using System.Diagnostics;
using Vernier;

int size    = 512;
int iters   = 50;
string backend = "cpu";

for (int i = 0; i + 1 < args.Length; i++)
{
    if (args[i] == "--size")    { size    = int.Parse(args[i + 1]); i++; }
    if (args[i] == "--iters")   { iters   = int.Parse(args[i + 1]); i++; }
    if (args[i] == "--backend") { backend = args[i + 1];            i++; }
}

Detector det;
try
{
    det = backend == "cuda" ? Detector.CreateCuda() : new Detector();
}
catch (InvalidOperationException ex)
{
    Console.Error.WriteLine($"cuda unavailable: {ex.Message}");
    Environment.Exit(2);
    return;
}

float[] img    = SyntheticImage(size);
float   period = size / 16.0f;

// Warm-up: prime FFT/GPU pipeline cache.
det.DetectPeriodic(img, size, size, period,
    sigma: 3.0f, minFrequency: 5, maxFrequency: 0, smoothingSigma: 0.5f);

double best  = double.PositiveInfinity;
double total = 0.0;
var sw = new Stopwatch();

for (int i = 0; i < iters; i++)
{
    sw.Restart();
    det.DetectPeriodic(img, size, size, period,
        sigma: 3.0f, minFrequency: 5, maxFrequency: 0, smoothingSigma: 0.5f);
    sw.Stop();
    double ms = sw.Elapsed.TotalMilliseconds;
    total += ms;
    if (ms < best) best = ms;
}

det.Dispose();

double mean = total / iters;
Console.WriteLine(
    $"binding=dotnet backend={backend} size={size}x{size} iters={iters} mean={mean:F2}ms best={best:F2}ms");


static float[] SyntheticImage(int size)
{
    const float TwoPi = MathF.PI * 2.0f;
    float period = size / 16.0f;
    float theta  = 0.1f;
    float cx = size / 2.0f, cy = size / 2.0f;
    float c = MathF.Cos(-theta), s = MathF.Sin(-theta);

    float[] img = new float[size * size];
    for (int row = 0; row < size; row++)
    {
        for (int col = 0; col < size; col++)
        {
            float dx = col - cx, dy = row - cy;
            float xp = c * dx - s * dy;
            img[row * size + col] = 0.5f + 0.5f * MathF.Cos(TwoPi * xp / period);
        }
    }
    return img;
}
