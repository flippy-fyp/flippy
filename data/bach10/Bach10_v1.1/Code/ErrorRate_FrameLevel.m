function Results = ErrorRate_FrameLevel(F0sEst, F0sReal)
% Evaluate the MPE results in each time frame.
% The estimated pitches and the ground-truth pitches should be in MIDI num.
% A pitch estimate is correct, if it deviates less than 0.5 semitone from
% its corresponding ground-truth pitch.
% There are three classes of measurements:
% 1. Predominant F0 Accuracy
%   Measure the first F0 estimate in each frame (usually the predominant F0
%   estimate). If it matches with any ground-truth F0s in that frame, it's correct.
% 2. Multiple F0 Accuracy
%   In each frame, each F0 estimate is tried to match with one ground-truth F0. A
%   ground-truth F0 can be matched to at most one F0 estimate.
% 3. Multiple F0 Accuracy without counting octave errors (Chroma Accuracy)
%   Convert F0 estimates and ground-truth pitches to pitch classes.
%   Then each pitch class estimate is tried to match with one ground-truth pitch class.
%   A ground-truth pitch class can be matched to at most one pitch class
%   estimate.
%
% Input:
%   - F0sEst: the estimated F0s, each column corresponds to a frame
%   - F0sReal: the real F0s, each column corresponds to a frame
% Output:
%   - Results
%       - Results.PreAcc : Predominant F0 Accuracy
%       - Results.MulPre : Multiple F0 Precision (TP/(TP+FP))
%       - Results.MulRec : Multiple F0 Recall (TP/(TP+FN))
%       - Results.MulAcc : Multiple F0 Accuracy (TP/(TP+FP+FN))
%       - Results.MulLowOctErrRate: Multiple F0 lower octave error rate
%           (LE/(TP+FP+FN))
%       - Results.MulUppOctErrRate: Multiple F0 upper octave error rate
%           (UE/(TP+FP+FN))
%       - Results.OctPre : Multiple F0 Octave Precision
%       - Results.OctRec : Multiple F0 Octave Recall
%       - Results.OctAcc : Multiple F0 Octave Accuracy
%       - Results.Etot   : Multiple F0 Error total
%       - Results.Esubs  : Multiple F0 Error substitution
%       - Results.Emiss  : Multiple F0 Error miss
%       - Results.Efa    : Multiple F0 Error false alarm
%       - Results.PolyphonyNo   : Polyphony histogram
%       - Results.Etot_step     : Error total (Etot) introduced in each iteration,
%           ONLY reasonable when the MPE algorithm is a iterative one,
%           the ground-truth and estimation contains only one frame, and the polyphony is given.
%       - Results.ErrorsIdx     : Indices of frames that have Multi-F0
%           estimation errors
%
% Author: Zhiyao Duan
% Created: 11/22/2008
% Last modified: 9/21/2009


nErrTot_step = zeros(1, size(F0sEst, 1));   % to record the number of errors in each iteration, ONLY reasonable when the MPE algorithm is a iterative one, the ground-truth and estimation contains only one frame, and the polyphony is given.
nCorrectPre = 0;                            % number of correct Predominant F0s
nCorrect = 0;                               % number of correct MultiF0 F0s
nCorrectOctave = 0;                         % number of correct MultiF0 F0s without counting octave errors
nPreReff = 0;                               % total number of reference pre-F0s
nTrans = 0;                                 % total number of transcribed (estimated) F0s
nReff = 0;                                  % total number of reference F0s
nErrTot = 0;                                % total number of errors
nErrSubs = 0;                               % number of substitution errors
nErrMiss = 0;                               % number of miss errors
nErrFa = 0;                                 % number of false alarm errors
nLowOctErr = 0;                             % number of lower octave errors
nUppOctErr = 0;                             % number of higher octave errors
Results.PolyphonyNo = zeros(1,10);          % number of frames with different estimated polyphony
Results.ErrorsIdx = [];                     % Indices of frames that have Multi-F0 errors

% begin process
FrameNo = size(F0sReal, 2);                 % number of total frames
for fnum = 1:FrameNo
    f0_est = F0sEst(:, fnum);
    f0_est = sort(f0_est(f0_est ~= 0));     % sorted by ascending order
    f0_est_octave = sort(mod(f0_est, 12));  % pitch classes, sorted by ascending order
    F0NumEst = length(f0_est);
    f0_real = F0sReal(:, fnum);
    f0_real = sort(f0_real(f0_real ~= 0));  % sorted by ascending order
    f0_real_octave = sort(mod(f0_real, 12));% sorted by ascending order
    F0NumReal = length(f0_real);

    % counts in the current frame
    nCurrCorrect = 0;
    nCurrLowOctErr = 0;
    nCurrUppOctErr = 0;
    nCurrCorrectOctave = 0;                 % withouth accounting octave errors

    % record the number of valid frames
    if F0NumReal ~= 0
        nPreReff = nPreReff + 1;
    end

    % No pitches are estimated, record errors and continue
    if F0NumEst == 0
    	nReff = nReff + F0NumReal;
        nErrTot = nErrTot + max(F0NumEst, F0NumReal) - nCurrCorrect;
        nErrSubs = nErrSubs + min(F0NumEst, F0NumReal) - nCurrCorrect;
        nErrMiss = nErrMiss + max(0, F0NumReal - F0NumEst);
        nErrFa = nErrFa + max(0, F0NumEst - F0NumReal);
        continue;
    end

    %PredominantF0
    for i=1:F0NumReal
        if (F0sEst(1, fnum) >= f0_real(i) - 0.5 && F0sEst(1, fnum) <= f0_real(i) + 0.5)
            nCorrectPre = nCorrectPre + 1;  % find a correct Predominant F0
            break;
        end
    end

    %MultipleF0
    isUsedEst = zeros(1,F0NumEst);              % record if an F0 estimate has been matched
    isUsedReal = zeros(1, F0NumReal);           % record if a reference F0 has been matched
    for i=1:F0NumReal                           % FOR each reference F0
        for j=1:F0NumEst                        % FOR each F0 estimate
            if (isUsedEst(j) == 0 ...
                && f0_est(j) >= f0_real(i) - 0.5 ...
                && f0_est(j) <= f0_real(i) + 0.5) % a match is found
                isUsedEst(j)=1;                 % this F0 estimate has been considered, cannot considered again
                isUsedReal(i)=1;                % this reference F0 has been considered, cannot considered again
                nCurrCorrect = nCurrCorrect+1;  % update the number of correctness
                break;
            end
        end
    end

    %MultipleF0 Octave Errors
    for i=1:F0NumReal                           % FOR each reference F0
        if isUsedReal(i) == 1                   % if this reference F0 has been used, then do not consider it again
            continue;
        end
        for j=1:F0NumEst                        % FOR each estimated F0
            if (isUsedEst(j) == 0 ...
                && (mod(f0_real(i) - f0_est(j), 12) <= 0.5 ...
                 || mod(f0_est(j) - f0_real(i), 12) <= 0.5))    % find a good match without considering octave errors
                if f0_est(j) < f0_real(i)
                    nCurrLowOctErr = nCurrLowOctErr + 1;
                    isUsedEst(j)=1;             % this F0 estimate has been considered, cannot be considered again
                    break;
                else
                    nCurrUppOctErr = nCurrUppOctErr + 1;
                    isUsedEst(j)=1;             % this F0 estimate has been considered, cannot be considered again
                    break;
                end
            end
        end
    end

    %MultipleF0 without counting octave errors (Chroma Accuracy)
    isUsedEst = zeros(1,F0NumEst);
    for i=1:F0NumReal                           % FOR each reference F0
        for j=1:F0NumEst                        % FOR each F0 estimate
            if (isUsedEst(j) == 0 ...
                && (mod(f0_est_octave(j) - f0_real_octave(i), 12) <= 0.5 ...
                 || mod(f0_real_octave(i) - f0_est_octave(j), 12) <= 0.5)) % find a good match without considering octave errors
                isUsedEst(j)=1;                 % this F0 estimate has been considered, cannot considered again
                nCurrCorrectOctave = nCurrCorrectOctave + 1;    % update the correctness without considering octave errors
                break;
            end
        end
    end

    % Record errors in each iteration (Suppose the MPE algorithm is an
    % iterative algorithm, which outputs F0s one by one).
    isUsedReal = zeros(1, F0NumReal);
    for j=1:F0NumEst
        bWrong = 1;                                 % flag to mark if this iteration introduces an error
        for i=1:F0NumReal
            if (isUsedReal(i) == 0 ...
                && f0_est(j) >= f0_real(i) - 0.5 ...
                && f0_est(j) <= f0_real(i) + 0.5)   % correct
                isUsedReal(i)=1;                    % this reference has been considered, cannot considered again
                bWrong = 0;
                break;
            end
        end
        if bWrong == 1
            nErrTot_step(j) = nErrTot_step(j) + 1;
        end
    end

    % Now we should have "nCurrCorrect + nCurrLowOctErr + nCurrUppOctErr = nCurrCorrectOctave"

    % Update counts
    nCorrect = nCorrect + nCurrCorrect;         % multiple F0 correct
    nLowOctErr = nLowOctErr + nCurrLowOctErr;
    nUppOctErr = nUppOctErr + nCurrUppOctErr;
    nCorrectOctave = nCorrectOctave + nCurrCorrectOctave; % multiple F0 Octave correct
    nTrans = nTrans + F0NumEst;                 % estimated
    nReff = nReff + F0NumReal;                  % reference
    nErrTot = nErrTot + max(F0NumEst, F0NumReal) - nCurrCorrect;
    nErrSubs = nErrSubs + min(F0NumEst, F0NumReal) - nCurrCorrect;
    nErrMiss = nErrMiss + max(0, F0NumReal - F0NumEst);
    nErrFa = nErrFa + max(0, F0NumEst - F0NumReal);

    % record error frames
    if nCurrCorrect ~= F0NumReal || nCurrCorrect ~= F0NumEst
        Results.ErrorsIdx = [Results.ErrorsIdx, fnum];
    end

    % record polyphony distribution
    if F0NumEst<9
        Results.PolyphonyNo(F0NumEst) = Results.PolyphonyNo(F0NumEst) + 1;
    else
        Results.PolyphonyNo(9) = Results.PolyphonyNo(9) + 1;
    end
end

Results.PreAcc = nCorrectPre / nPreReff;                % Predominant F0 Accuracy
Results.MulPre = nCorrect / nTrans;                     % Multi-F0 Precision
Results.MulRec = nCorrect / nReff;                      % Multi-F0 Recall
Results.MulAcc = nCorrect / (nTrans + nReff - nCorrect);% Multi-F0 Accuracy: TP/(TP+FP+FN)
Results.MulLowOctErrRate = nLowOctErr / (nTrans + nReff - nCorrect);    % Lower octave error rate
Results.MulUppOctErrRate = nUppOctErr / (nTrans + nReff - nCorrect);    % higher octave error rate
Results.OctPre = nCorrectOctave / nTrans;               % Multi-Pitch-Class errors Precision
Results.OctRec = nCorrectOctave / nReff;                % Multi-Pitch-Class Recall
Results.OctAcc = nCorrectOctave / (nTrans + nReff - nCorrectOctave);    % Multi-F0 without counting octave errors Accuracy
Results.Etot = nErrTot / nReff;                         % Multi-F0 Error_total
Results.Esubs = nErrSubs / nReff;                       % Error_substitution
Results.Emiss = nErrMiss / nReff;                       % Error_miss
Results.Efa = nErrFa / nReff;                           % Error_falsealarm
Results.Etot_step = nErrTot_step / nReff;               % Error introduced in each iteration of the MPE algorithm
