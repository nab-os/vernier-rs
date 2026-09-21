# vernier-webapp

A browser front-end for `vernier-patterns`, built with [Dioxus](https://dioxuslabs.com).
Pick a generator, tune every parameter it exposes, and see the render update
live.

The patterns are produced by `vernier-patterns` itself, compiled to
WebAssembly — the same code the CLI and the round-trip tests call. This crate
only wraps the parameters in widgets and blits the resulting intensity field
onto a `<canvas>`, so what the page shows is what the library generates.

## Running it

```sh
rustup target add wasm32-unknown-unknown
cargo install dioxus-cli@0.7.10 --locked   # provides `dx`

cd vernier-webapp
dx serve --release                         # http://localhost:8080
```

`dx bundle --release` produces a static directory that can be served from
anywhere; there is no server side.

**Use `--release`.** The generators are per-pixel CPU loops, so the build
profile decides whether the preview keeps up with the slider. Measured in
headless Chromium on this machine, per render:

| Image | debug (`opt-level = 2`) | release |
| --- | --- | --- |
| 256² | 137 ms | 14 ms |
| 512² | 471 ms | 53 ms |
| 1024² | 1.4 s | 223 ms |
| 2048² | 5.5 s | 852 ms |

Release is interactive up to 1024²; 2048² is a "set it and download it" size
either way. The dev profile is already pinned to `opt-level = 2` in
`Cargo.toml` because a fully unoptimized wasm build takes ~960 ms on a 512²
render, which makes dragging a slider feel broken.

Most of that time is the per-pixel closure in `vernier_patterns::render`, which
recomputes `sin_cos` of the pose angle for every pixel rather than hoisting it
out of the loop. Hoisting it would speed up the CLI and the test suite as much
as this page, but it is a change to the library, not to this crate, so it is
left alone here.

## The two views

The header switches between them, and they share the pattern selection.

**Generator** renders a pattern and hands it back as a PNG.

**Spectrum explorer** puts a virtual camera in front of that pattern and shows
every stage the detector passes through as you move it:

    camera image -> FFT -> peak selection -> band-pass -> reconstruction -> phases

Drag the camera image: left to translate, right to rotate about Z and change
distance, middle to tilt out of plane, the wheel for distance, shift for finer
motion. The detector is the library itself — `vernier-spectral` and
`vernier-cpu` compiled to WebAssembly — running in the page, with no server.

Two things worth knowing:

- **Stamp** and **QR-like** are stubs with no layout to sample at a pose, so the
  explorer refuses them. The other three work.
- **Log scale on the FFT panels** is on by default: a coded pattern's carrier
  peaks stand orders of magnitude above its sidebands, so a linear ramp shows
  two white dots on black. Linear is the honest view of the magnitudes.

Out-of-plane tilt is the freedom `PatternPose` cannot express, so the explorer
keeps its own six-freedom pose and does the projection in `camera.rs`.

## What you can set

| Section | Parameters |
| --- | --- |
| Pattern type | Periodic, Megarena, Checkerboard, Stamp, QR-like |
| Image | width, height (16–2048 px), plus 256/512/1024/2048 presets |
| Periodic | spatial period (px) |
| Megarena | dot period (px), LFSR order (4–12 bits), LFSR offset |
| Checkerboard | code layout (squares or diamonds), square side (px), LFSR order (4–12 bits), LFSR offset, supersampling (1–8×/edge), uncoded reference |
| Stamp | tile size (px) |
| QR-like | modules per axis, module size (px) |
| Pose | X translation, Y translation, orientation (degrees, shown in radians too) |
| Display | invert, actual size vs. scale-to-fit |

Sliders commit continuously so a parameter can be swept while watching the
pattern move; the number box beside each one commits on blur or Enter for an
exact value. **Download PNG** saves the render at full resolution regardless of
the preview scale, and the *Equivalent vernier-patterns call* panel prints the
Rust that reproduces the current settings, so a parameter set found here can be
pasted straight into a test.

Alongside the preview, the readouts show what the parameters work out to — for
Megarena, the code length (`2^order − 1`) and the absolute range in pixels
(`code_length × 3 × period`, three periods per bit).

## Three parameters the checkerboard needs care with

**Code layout** decides which way the coded lattice sits. `Squares` writes the
code along the square edges, leaving the carriers on the diagonals (peaks at
±45°); `Diamonds` turns the lattice 45°, putting the carriers on the pattern
axes (peaks at 0° and 90°) for a factor of √2 in absolute range. Same carrier
period either way.


**Square side is not the carrier period.** The checkerboard's two carriers run
along the diagonals, at ±45° to the square edges, so successive fringes are
`a·√2` apart. A detector configured with the square side would be wrong by that
factor, so the panel shows the derived carrier period in both the field hint and
the readouts, next to the side you actually set.

**Supersampling is not cosmetic.** Unlike the other generators the checkerboard
is binary with hard edges, so point sampling (`1×`) aliases, and the aliasing
moves the *measured carrier phase* — it corrupts the measurement, not just the
picture. The control goes down to `1×` because seeing that happen is useful;
`4×` is the upstream default and the right setting for generating real targets.

It is also the one control here that costs real time, since it multiplies the
per-pixel work by its square. Release build, 512², best of five:

| Supersampling | Samples/px | Per render |
| --- | --- | --- |
| 1× | 1 | 26 ms |
| 2× | 4 | 68 ms |
| 4× (default) | 16 | 237 ms |
| 8× | 64 | 866 ms |

So the default checkerboard costs about 4× a megarena of the same size, and `8×`
is slow enough to feel. Sweep a slider at `1×` or `2×` if you are hunting for a
parameter set, then put it back to `4×` for the render you keep.

The **uncoded reference** toggle renders the plain checkerboard with every
coding site left at its parity colour. Flipping it against the coded render
shows exactly which squares the position code inverts — roughly one square in
nine.

## Stubs

`Stamp` and `QrLike` render a blank field: `vernier-patterns` fixes their
interfaces but leaves the layouts unimplemented, deliberately, rather than
guessing at an encoding. Their controls are wired to the real constructor
parameters, and the UI says so on screen instead of showing an unexplained black
square. When `vernier-patterns/src/stamp.rs` and `src/qrcode.rs` gain their
rasterizers, this app picks them up with no change here.

## Layout

| File | Contents |
| --- | --- |
| `src/pattern.rs` | the parameter model and its mapping onto `vernier-patterns` calls |
| `src/canvas.rs` | canvas painting and PNG download |
| `src/controls.rs` | the slider/number, toggle and section widgets |
| `src/main.rs` | the app shell, the view switch, and the per-generator field lists |
| `src/camera.rs` | the explorer's six-freedom pose and its projection |
| `src/spectral.rs` | one pass of the detector, keeping every stage |
| `src/explorer.rs` | the explorer view |

Adding a parameter means a field on `PatternSettings`, an arm in
`PatternSettings::render`, and a `NumberField` in `pattern_fields` — nothing
else is wired per-parameter.

## Why it is its own workspace

`vernier-webapp/Cargo.toml` carries an empty `[workspace]` table, so it is not a
member of the root workspace. The root workspace builds for the host and this
crate only makes sense on `wasm32`; keeping it separate means `cargo build` at
the repo root never has to compile a web renderer it cannot run.
