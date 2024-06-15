% basic process; generate white noise, filter it, epoch it to make smaller
% impulses, generate a sequence of impulses.

% generate white noise of 1s duration:
fs = 44100;
r = randn(1,fs);
plot(r); 
%soundsc(r,fs)% sounds suitably horrible.

% now define filter properties; let's generate 1 octave wide bands
% first one is 2-4 k second one will be  8-16k note these values are
% selected so that ITD cues are negligable (since we aren't feeding them
% into the model anyway)
fL = 2000;fH = 4000;
hf = design(fdesign.bandpass('N,F3dB1,F3dB2',10,fL,fH,fs));
y1 = filter(hf,r);

% play it as a sanity check:
%soundsc(y1,fs)% 


fL = 8000;fH = 16000;
hf = design(fdesign.bandpass('N,F3dB1,F3dB2',10,fL,fH,fs));
y2 = filter(hf,r);
% play it as a sanity check:
%soundsc(y2,fs)% 


% an FFT of these sounds should yield something with a clear peak in the
% frequency domain!

%now let's make some impulses... since our bin width is 20 ms there isn't
%much point in doing anything shorter than that...

impDurL = round(fs*0.04);% 50 ms 
impDurH = round(fs*0.03);% 40 ms 

chunkL = y1(10001:10001+impDurL);% pick a random chunk from the middle (filtering may do weird things to the ends)
chunkH = y2(10001:10001+impDurH);

% normalise to RMS
chunkL = chunkL.*(sqrt(mean((chunkL).^2)));
chunkH = chunkH.*(sqrt(mean((chunkH).^2)));

subplot(3,2,1);
plot(chunkL)
subplot(3,2,2);
plot(chunkH)

impL = envelopeEnds(chunkL,0.007,fs); % give slightly differnet onset ramps as this will facilitate grouping
impH = envelopeEnds(chunkH,0.005,fs);

subplot(3,2,3)
plot(impL)
subplot(3,2,4)
plot(impH);

% now generate the impulse train; make 1s chunks that can be repeated
rate1 = 3; %2Hz presentation
rate2 = 5;

seq1 = zeros(1,fs);
seq2 = zeros(1,fs);

% let's put the 2Hz in at 0.25 and 0.75 s
for ss = 0.001:1/rate1:1
startSamp = round(ss*fs);
seq1(startSamp:startSamp+impDurL) = impL;
end
subplot(3,2,5) 
plot(seq1);

% let's put the 2Hz in at 0.25 and 0.75 s
for ss = 0.001:1/rate2:1
startSamp = round(ss*fs);
seq2(startSamp:startSamp+impDurH) = impH;
end
subplot(3,2,6) 
plot(seq2);

audiowrite('low.wav',seq1,fs)

% high file is clipping so reduce amplitude; 
seq2 = seq2./max(abs(seq2));
audiowrite('high.wav',seq2,fs)


function output_signal = envelopeEnds(signal,ramp,fs)
% This function tries to remove the transients in the signal by enveloping
% a ramp (in seconds) 
% envelope(signal,fs,ramp)

samples = ceil(ramp*fs); % added ceil to make this an integer
x = -pi:pi/samples:0;
y = 0:pi/samples:pi;
output_signal = signal;


% prepare the envelope functions
envelope_function(1:samples) = cos(x(1:samples))/2+0.5;

% fade in
for i = 1 : samples
    output_signal(i) = signal(i) * envelope_function(i);
end

% fade out
for i = 0 : (samples-1)
    current_position = length(signal) - i;
    output_signal(current_position) = signal(current_position) * envelope_function(i+1);
end
end



