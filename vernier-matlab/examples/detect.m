% detect.m — vernier MATLAB quick-start example.
%
% Loads the reference 9 µm / 12-bit Megarena image and runs both periodic
% and absolute detection.  Expected output (matches vernier-cli):
%
%   megarena  x = -4452.12   y = -26759.51   theta = 1.4295
%
% Usage
% -----
%   % 1. Add the +vernier package to the MATLAB path:
%   addpath(fullfile(fileparts(mfilename('fullpath')), '..'))
%
%   % 2. Point to the native library if not in the default location:
%   vernier.Detector.set_lib_path('/path/to/libvernier_cabi.so')
%
%   % 3. Run:
%   detect

% ── Setup ─────────────────────────────────────────────────────────────────────

here    = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));          % puts +vernier on the path

img_path = fullfile(here, '..', '..', 'resources', 'images', ...
    'megarenaPatternImage_12bits_9um.jpg');

% ── Load image ────────────────────────────────────────────────────────────────

raw = imread(img_path);                % uint8 [H x W] or [H x W x 3]
if ndims(raw) == 3                     %#ok<ISMAT>
    raw = rgb2gray(raw);
end
img = single(raw) / 255;              % float32 [H x W], values in [0, 1]
fprintf('image   : %d×%d  (%s)\n', size(img,2), size(img,1), ...
    'megarenaPatternImage_12bits_9um.jpg');

% ── Detector ──────────────────────────────────────────────────────────────────

det = vernier.Detector();
fprintf('backend : cpu-rustfft\n');

% ── Periodic detection ────────────────────────────────────────────────────────

pose = det.detect_periodic(img, 9.0, ...
    'min_frequency', 20, 'max_frequency', 500);
fprintf('periodic  x=%.4f  y=%.4f  theta=%.6f\n', pose.x, pose.y, pose.theta);

% ── Megarena absolute detection ───────────────────────────────────────────────

pose = det.detect_megarena(img, 9.0, 12, ...
    'min_frequency', 20, 'max_frequency', 500);
fprintf('megarena  x=%.2f  y=%.2f  theta=%.6f\n', pose.x, pose.y, pose.theta);
fprintf('expected  x≈-4452.12  y≈-26759.51  (matches vernier-cli)\n');

% ── CUDA (graceful fallback) ──────────────────────────────────────────────────

try
    gpu  = vernier.Detector.cuda();
    pg   = gpu.detect_megarena(img, 9.0, 12, ...
        'min_frequency', 20, 'max_frequency', 500);
    fprintf('cuda      x=%.2f  y=%.2f\n', pg.x, pg.y);
catch e
    fprintf('cuda      not available: %s\n', e.message);
end
