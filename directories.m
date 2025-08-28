folderPath = '/media/niklas/SSD2/Dataset/Dipco/audio/dev';
% Define the folder path

% Define the folder path


% Define the regular expression pattern for the file name
pattern = '^S(\d{2})_U(\d{2})\.CH(\d)\.wav$';

% Get a list of all files in the folder
fileList = dir(fullfile(folderPath, '*.wav'));

% Initialize cell arrays for the three groups
group1 = {};  % Group [1,7,4]
group2 = {};  % Group [2,7,5]
group3 = {};  % Group [6,7,3]

% Loop through each file and check if it matches the pattern
for i = 1:length(fileList)
    % Match the file name with the regular expression pattern
    tokens = regexp(fileList(i).name, pattern, 'tokens');
    
    % If a match is found
    if ~isempty(tokens)
        % Extract the channel number (from the third token)
        channel = str2double(tokens{1}{3});
        
        % Group the file based on the channel number
        if ismember(channel, [1, 7, 4])
            group1{end+1} = fileList(i).name;  % Add to group1
        elseif ismember(channel, [2, 7, 5])
            group2{end+1} = fileList(i).name;  % Add to group2
        elseif ismember(channel, [6, 7, 3])
            group3{end+1} = fileList(i).name;  % Add to group3
        end
    end
end

% Display the grouped files
disp('Group [1,7,4]:');
%disp(group1);

disp('Group [2,7,5]:');
%disp(group2);

disp('Group [6,7,3]:');
%disp(group3);
% Example groups

% Example strings
strings = group1;

% Initialize a container for grouped strings
groups = struct();

% Loop through each string
for i = 1:length(strings)
    % Extract the prefix using a regular expression (everything before "_X")
    prefix = regexp(strings{i}, '^(S\d+_U\d+)', 'match', 'once');
    
    if ~isempty(prefix)
        % If the prefix exists in the structure, append the string
        if isfield(groups, prefix)
            groups.(prefix){end+1} = strings{i};
        else
            % Otherwise, create a new field with this prefix
            groups.(prefix) = {strings{i}};
        end
    end
end

% Display the grouped strings
disp('Grouped strings:');
groupNames = fieldnames(groups);
prefixPath = folderPath;

% Apply the function to all fields of the structure
groups = structfun(@(group) fullfile(prefixPath, group), groups, 'UniformOutput', false);
groupNames = fieldnames(groups); % Get the group names (e.g., S01_U01, S02_U02)
for i = 1:length(groupNames)
    % Extract the group and its file paths
    groupFiles = groups.(groupNames{i});
    
    % Use the first file name in the group (without path) as the output file name
    [~, baseName, ~] = fileparts(groupFiles{1}); % Extract base name from the first file path
    outputFileName = sprintf('%s_beamformed.wav', baseName); % Generate output file name
    
    % Call the beamforming function
    fprintf('Processing group %s...\n', groupNames{i});
    beamformGroup(groupFiles, outputFileName);
end

function beamformGroup(filePaths, outputFileName)
    % BEAMFORMGROUP Perform beamforming on a group of audio files.
    % Inputs:
    %   filePaths - Cell array of file paths to audio files for beamforming.
    %   outputFileName - Name of the output file for the beamformed signal.
    
    % Step 1: Validate Inputs
    if isempty(filePaths)
        error('No file paths provided.');
    end
    if nargin < 2 || isempty(outputFileName)
        error('Output file name must be specified.');
    end

    % Step 2: Initialize Variables
    nChannels = length(filePaths); % Number of channels
    audioData = [];
    fs = 0;

    % Step 3: Read Audio Data from Files
    for i = 1:nChannels
        [data, currentFs] = audioread(filePaths{i});
        if i == 1
            fs = currentFs; % Get sampling rate from the first file
            nSamples = size(data, 1);
            audioData = zeros(nSamples, nChannels); % Preallocate matrix
        elseif currentFs ~= fs
            error('Sampling rate mismatch between files.');
        end
        audioData(:, i) = data; % Add the data as a column
    end

    % Step 4: Define Microphone Array Geometry
    micSpacing = 0.035; % Microphone spacing in meters
    array = phased.ULA('NumElements', nChannels, 'ElementSpacing', micSpacing);
    c = 343; % Speed of sound in m/s

    % Step 5: Create the Beamformer
    targetDirection = [0; 0]; % Target azimuth and elevation (broadside)
    beamformer = phased.TimeDelayBeamformer('SensorArray', array, ...
        'SampleRate', fs, ...
        'PropagationSpeed', c, ...
        'Direction', targetDirection);

    % Step 6: Apply Beamforming
    beamformedSignal = beamformer(audioData);

    % Step 7: Save the Beamformed Signal
    audiowrite(outputFileName, beamformedSignal, fs);

    % Display Success Message
    fprintf('Beamformed signal saved as "%s"\n', outputFileName);
end
