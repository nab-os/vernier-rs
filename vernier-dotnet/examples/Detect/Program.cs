// vernier-dotnet quick-start example.
//
// Generates a 512×512 synthetic sinusoidal periodic image and runs
// DetectPeriodic.  The recovered sub-period phase should be near zero because
// the pattern is centred at the origin.
//
// Build and run:
//   cargo build -p vernier-cabi --release          # build native library
//   LD_LIBRARY_PATH=<repo>/target/release \
//       dotnet run --project vernier-dotnet/examples/Detect/Detect.csproj

using System;
using Vernier;

const int W = 512;
const int H = 512;
const float Period = 40.0f;
const float TwoPi  = MathF.PI * 2.0f;

// Synthetic image: cos(2π·col/T) × cos(2π·row/T) mapped to [0, 1].
float[] pixels = new float[W * H];
for (int row = 0; row < H; ++row)
    for (int col = 0; col < W; ++col)
        pixels[row * W + col] =
            0.5f * (1.0f + MathF.Cos(TwoPi * col / Period))
          * 0.5f * (1.0f + MathF.Cos(TwoPi * row / Period));

// CPU detector ---------------------------------------------------------------
using var det = new Detector();

Pose p = det.DetectPeriodic(pixels, W, H, Period,
    sigma: 3.0f, minFrequency: 5, maxFrequency: 0, smoothingSigma: 0.5f);

Console.WriteLine($"periodic  x={p.X:F4}  y={p.Y:F4}  theta={p.Theta:F6} rad");

// CUDA (graceful fallback) ---------------------------------------------------
try
{
    using var gpu  = Detector.CreateCuda();
    Pose pg = gpu.DetectPeriodic(pixels, W, H, Period,
        sigma: 3.0f, minFrequency: 5, maxFrequency: 0, smoothingSigma: 0.5f);
    Console.WriteLine($"cuda      x={pg.X:F4}  y={pg.Y:F4}");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"cuda      not available: {ex.Message}");
}
