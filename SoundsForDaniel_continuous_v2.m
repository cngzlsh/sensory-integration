% Basic process; generate white noise, filter it, epoch it to make smaller
% impulses, generate a sequence of impulses.

% Generate white noise of 1s duration:
fs = 44100;               % Sampling frequency
duration = 1;             % Duration in seconds
r = randn(1, fs * duration);  % Generate white noise
r = r * 0.01;             % Significantly reduce the amplitude of the white noise
figure;
subplot(4, 2, 1);
plot(r);
title('White Noise');
%soundsc(r, fs)           % Play the white noise as a sanity check

% Define filter properties; generate 1 octave wide bands
% First band is 2-4 kHz, second one is 8-16 kHz
fL1 = 2000; fH1 = 4000;
bpFilt1 = designfilt('bandpassiir', 'FilterOrder', 10, ...
                     'HalfPowerFrequency1', fL1, 'HalfPowerFrequency2', fH1, ...
                     'SampleRate', fs);
y1 = filter(bpFilt1, r);

fL2 = 8000; fH2 = 16000;
bpFilt2 = designfilt('bandpassiir', 'FilterOrder', 10, ...
                     'HalfPowerFrequency1', fL2, 'HalfPowerFrequency2', fH2, ...
                     'SampleRate', fs);
y2 = filter(bpFilt2, r);

% Plot filtered signals for verification
subplot(4, 2, 2);
plot(y1);
title('Filtered Noise (2-4 kHz)');

subplot(4, 2, 3);
plot(y2);
title('Filtered Noise (8-16 kHz)');

% Generate chunks for impulses
impDurL = length(y1); % Entire duration (1 second)
impDurH = length(y2); % Entire duration (1 second)

chunkL = y1; % Use the entire filtered signal for the impulse
chunkH = y2;

% Normalize to RMS and increase amplitude
chunkL = chunkL / rms(chunkL) * 10;
chunkH = chunkH / rms(chunkH) * 10;

subplot(4, 2, 4);
plot(chunkL);
title('Chunk L (2-4 kHz)');

subplot(4, 2, 5);
plot(chunkH);
title('Chunk H (8-16 kHz)');

impL = envelopeEnds(chunkL, 0.007, fs); % Apply envelope with 7 ms ramp
impH = envelopeEnds(chunkH, 0.005, fs); % Apply envelope with 5 ms ramp

subplot(4, 2, 6);
plot(impL);
title('Enveloped Impulse L (2-4 kHz)');

subplot(4, 2, 7);
plot(impH);
title('Enveloped Impulse H (8-16 kHz)');

% Generate impulse trains and superimpose on white noise
seq1 = r + impL; % Superimpose the impulse on the white noise
seq2 = r + impH;

subplot(4, 2, 8);
plot(seq1);
title('Impulse Train L (2-4 kHz)');

figure;
plot(seq2);
title('Impulse Train H (8-16 kHz)');

audiowrite('low_with_noise_v2.wav', seq1, fs);
audiowrite('high_with_noise_v2.wav', seq2, fs);

% Envelope function
function output_signal = envelopeEnds(signal, ramp, fs)
    % This function tries to remove the transients in the signal by enveloping
    % a ramp (in seconds)

    samples = ceil(ramp * fs); % added ceil to make this an integer
    x = -pi : pi / samples : 0;
    y = 0 : pi / samples : pi;
    output_signal = signal;

    % Prepare the envelope functions
    envelope_function = cos(x(1:samples)) / 2 + 0.5;

    % Fade in
    for i = 1 : samples
        output_signal(i) = signal(i) * envelope_function(i);
    end

    % Fade out
    for i = 0 : (samples - 1)
        current_position = length(signal) - i;
        output_signal(current_position) = signal(current_position) * envelope_function(i + 1);
    end
end