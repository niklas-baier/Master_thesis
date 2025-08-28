% Quick verification script for beamforming results
function verify_beamforming_results()

%% Load and compare signals
fprintf('Loading signals for comparison...\n');

% Load original (first channel)
original_file = '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH1.wav';
[original, fs] = audioread(original_file);

% Load beamformed result
beamformed_file = 'beamformed_for_ASR.wav';
if exist(beamformed_file, 'file')
    [beamformed, fs_beam] = audioread(beamformed_file);
    
    if fs ~= fs_beam
        error('Sample rate mismatch between files');
    end
else
    error('Beamformed file not found. Make sure beamforming completed successfully.');
end

% Make sure both signals have the same length
minLen = min(length(original), length(beamformed));
original = original(1:minLen);
beamformed = beamformed(1:minLen);

%% Basic quality metrics
fprintf('\n=== QUALITY COMPARISON ===\n');

% RMS values
rms_original = sqrt(mean(original.^2));
rms_beamformed = sqrt(mean(beamformed.^2));
fprintf('RMS - Original: %.4f, Beamformed: %.4f\n', rms_original, rms_beamformed);

% Peak values
peak_original = max(abs(original));
peak_beamformed = max(abs(beamformed));
fprintf('Peak - Original: %.4f, Beamformed: %.4f\n', peak_original, peak_beamformed);

% Dynamic range
dr_original = 20*log10(peak_original/rms_original);
dr_beamformed = 20*log10(peak_beamformed/rms_beamformed);
fprintf('Dynamic Range - Original: %.1f dB, Beamformed: %.1f dB\n', dr_original, dr_beamformed);

%% Spectral analysis
fprintf('\n=== SPECTRAL ANALYSIS ===\n');

% Compute power spectral density
[Pxx_orig, f] = pwelch(original, hamming(2048), 1024, 2048, fs);
[Pxx_beam, ~] = pwelch(beamformed, hamming(2048), 1024, 2048, fs);

% Focus on speech frequencies (300-3400 Hz)
speech_idx = f >= 300 & f <= 3400;
speech_power_orig = mean(Pxx_orig(speech_idx));
speech_power_beam = mean(Pxx_beam(speech_idx));

% Focus on noise frequencies (below 300 Hz and above 3400 Hz)
noise_idx = f < 300 | f > 3400;
noise_power_orig = mean(Pxx_orig(noise_idx));
noise_power_beam = mean(Pxx_beam(noise_idx));

fprintf('Speech band power - Original: %.2e, Beamformed: %.2e\n', ...
    speech_power_orig, speech_power_beam);
fprintf('Noise band power - Original: %.2e, Beamformed: %.2e\n', ...
    noise_power_orig, noise_power_beam);

% Calculate speech-to-noise ratio in frequency domain
snr_spectral_orig = 10*log10(speech_power_orig/noise_power_orig);
snr_spectral_beam = 10*log10(speech_power_beam/noise_power_beam);
fprintf('Spectral SNR - Original: %.1f dB, Beamformed: %.1f dB\n', ...
    snr_spectral_orig, snr_spectral_beam);
fprintf('Spectral SNR improvement: %.1f dB\n', snr_spectral_beam - snr_spectral_orig);

%% Cross-correlation analysis
fprintf('\n=== SPATIAL PROCESSING VERIFICATION ===\n');

% Check if beamformed signal is actually different from original
correlation = corrcoef(original, beamformed);
fprintf('Correlation between original and beamformed: %.3f\n', correlation(1,2));

if correlation(1,2) > 0.99
    fprintf('⚠️  WARNING: Signals are very similar. Beamforming may not be effective.\n');
elseif correlation(1,2) > 0.8
    fprintf('✓ Signals are related but different - good beamforming.\n');
else
    fprintf('⚠️  Signals are quite different - verify processing.\n');
end

%% Check for multiple channels to verify array processing
fprintf('\n=== ARRAY VERIFICATION ===\n');

% Try to load all channels for comparison
filePaths = {
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH1.wav',
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH7.wav',
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/S03_U01.CH4.wav'
};

channel_correlations = [];
for i = 1:length(filePaths)
    if exist(filePaths{i}, 'file')
        [ch_data, ~] = audioread(filePaths{i});
        ch_data = ch_data(1:minLen);
        
        % Correlation with beamformed output
        corr_ch = corrcoef(ch_data, beamformed);
        channel_correlations(i) = corr_ch(1,2);
        
        fprintf('Channel %d correlation with beamformed: %.3f\n', i, corr_ch(1,2));
    end
end

% Check if beamforming actually combined channels
if length(channel_correlations) > 1
    max_single_corr = max(channel_correlations);
    if max_single_corr < 0.98
        fprintf('✓ Beamformed signal differs from individual channels - array processing worked.\n');
    else
        fprintf('⚠️  Beamformed signal very similar to one channel - limited array benefit.\n');
    end
end

%% Recommendations for ASR
fprintf('\n=== ASR RECOMMENDATIONS ===\n');

if snr_spectral_beam > snr_spectral_orig + 1
    fprintf('✓ Good spectral improvement for ASR\n');
elseif correlation(1,2) < 0.95
    fprintf('✓ Signal processing applied, may still benefit ASR through noise reduction\n');
else
    fprintf('⚠️  Limited improvement detected. Consider:\n');
    fprintf('   - Adjusting VAD parameters\n');
    fprintf('   - Using different target direction\n');
    fprintf('   - Checking if environment has directional noise\n');
end

% File size and duration info
info_orig = audioinfo(original_file);
info_beam = audioinfo(beamformed_file);
fprintf('\nFile info:\n');
fprintf('Original: %.1fs\n', info_orig.Duration);
fprintf('Beamformed: %.1fs\n', info_beam.Duration);

% Get file sizes using dir
orig_dir = dir(original_file);
beam_dir = dir(beamformed_file);
if ~isempty(orig_dir) && ~isempty(beam_dir)
    fprintf('File sizes - Original: %.1fMB, Beamformed: %.1fMB\n', ...
        orig_dir.bytes/1e6, beam_dir.bytes/1e6);
end

fprintf('\n=== TESTING WITH WHISPER ===\n');
fprintf('Your beamformed file is ready for ASR testing:\n');
fprintf('File: %s\n', beamformed_file);
fprintf('Next steps:\n');
fprintf('1. Test with your Whisper model\n');
fprintf('2. Compare ASR accuracy: original vs beamformed\n');
fprintf('3. If no improvement, try adjusting beamforming parameters\n');

end