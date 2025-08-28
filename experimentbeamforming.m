% Inputs
nChannels = 2; % Number of microphone channels
%azimuth for s3u3 => -17 S3u2 => -2 S3u1 => -17 ? direction matters 17 =>
%-17 u04 => -13  u05 => 12 17 
channelFiles = {  '/media/niklas/SSD2/Dataset/Dipco/audio/eval/output9.wav', ... % Replace with actual file paths
    '/media/niklas/SSD2/Dataset/Dipco/audio/eval/output10.wav','/media/niklas/SSD2/Dataset/Dipco/audio/eval/output11.wav',}; % List of .wav files for each channel
outputFile_LCMV = 'beamformed_LCMV.wav';
outputFile_GSC = 'beamformed_GSC.wav';
micSpacing = 0.035; % Microphone spacing in meters
fs = 16000; % Sampling frequency (Hz)
c = 343; % Speed of sound (m/s)

% Step 1: Load multichannel data
disp('Loading multichannel data...');
audioData = []; % Initialize an empty array for audio data
for i = 1:nChannels
    [channelData, fsChannel] = audioread(channelFiles{i});
    if i == 1
        fs = fsChannel; % Set sampling frequency from the first channel
    elseif fs ~= fsChannel
        error('Sampling frequencies of input channels do not match!');
    end
    audioData = [audioData, channelData]; % Concatenate data into multichannel format
end

% Create microphone array
array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);

% Step 2: Detect Optimal Direction
disp('Scanning for optimal direction...');
azimuthRange = -90:1:90; % Scan azimuth angles
elevationRange = 0; % Assume flat plane (no elevation changes)
maxPower = -Inf;
optimalDirection = [0; 0];

% Scan the azimuth range
for az = azimuthRange
    targetDirection = [az; elevationRange]; % Direction [Azimuth; Elevation]
    % Create time-delay beamformer for scanning
    scanner = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, ...
        'PropagationSpeed', c, ...
        'Direction', targetDirection);
    % Apply beamforming
    scannedSignal = scanner(audioData);
    % Measure power
    signalPower = sum(scannedSignal.^2, 'all');
    if signalPower > maxPower
        maxPower = signalPower;
        optimalDirection = targetDirection;
    end
end

disp(['Optimal direction detected: Azimuth = ', num2str(optimalDirection(1)), ...
    ', Elevation = ', num2str(optimalDirection(2))]);

% Step 3: LCMV Beamforming
disp('Applying LCMV Beamformer...');
beamformer_LCMV = phased.LCMVBeamformer('SensorArray', array, ...
    'PropagationSpeed', c, ...
    'SampleRate', fs, ...
    'Direction', optimalDirection, ...
    'Constraint', eye(nChannels)); % Constrain to preserve target signal
beamformedSignal_LCMV = beamformer_LCMV(audioData);

% Normalize the beamformed LCMV signal
beamformedSignal_LCMV = beamformedSignal_LCMV / max(abs(beamformedSignal_LCMV));

% Save LCMV beamformed signal
audiowrite(outputFile_LCMV, beamformedSignal_LCMV, fs);
disp(['LCMV beamformed signal saved as "', outputFile_LCMV, '"']);

% Step 4: GSC Beamforming
disp('Applying GSC Beamformer...');
beamformer_GSC = phased.GSCBeamformer('SensorArray', array, ...
    'PropagationSpeed', c, ...
    'SampleRate', fs, ...
    'Direction', optimalDirection, ...
    'BlockingMatrix', phased.GSCBlockingMatrix('SensorArray', array));
beamformedSignal_GSC = beamformer_GSC(audioData);

% Normalize the beamformed GSC signal
beamformedSignal_GSC = beamformedSignal_GSC / max(abs(beamformedSignal_GSC));

% Save GSC beamformed signal
audiowrite(outputFile_GSC, beamformedSignal_GSC, fs);
disp(['GSC beamformed signal saved as "', outputFile_GSC, '"']);
