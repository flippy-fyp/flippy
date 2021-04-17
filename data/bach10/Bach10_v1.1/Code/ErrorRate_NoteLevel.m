function Results_note = ErrorRate_NoteLevel(EstNote, GTNote, para)
% Note-level error rate calculation. Compare a list of estimated notes with
% a list of ground-truth notes.
%
% Input
%   - EstNote       : estimated notes, each cell corresponds to a note: first line is frame number,
%                     second line is frequency in MIDI number
%   - GTNote        : ground-truth notes, the same format as EstNote
%   - para
%       - bUseOffset: the flag that if using offset criterion or not
%       - OnsetDiff : the onset time difference criterion (in frame number)
%       - OffsetDiff: the offset time difference criterion (in ratio of the length of reference length)
% Output
%   - Results_note  : Results
%       - Pre       : Precision
%       - Rec       : Recall
%       - Fme       : F-measure
%       - AOR       : Average overlap ratio
%       - OctPre    : Precision without considering octave errors
%       - OctRec    : Recall without considering octave errors
%       - OctFme    : F-measure without considering octave errors
%       - OctAOR    : Average overlap ratio without considering octave errors
%
% Author: Zhiyao Duan
% Created: 5/5/2009
% Last modified: 9/22/2009

bUseOffset = para.bUseOffset;
OnsetDiff = para.OnsetDiff;
OffsetDiff = para.OffsetDiff;

EstNoteNum = length(EstNote);                       % number of estimated notes
GTNoteNum = length(GTNote);                         % number of ground-truth notes

nCorrect = 0;                                       % number of correctly estimated notes
AOR = 0;                                            % average overlap ratio between estimated and ground-truth notes
nChromaCorrect = 0;                                 % number of correctly estimated notes, without counting octave errors
ChromaAOR = 0;                                      % average overlap ratio between estimated and ground-truth notes, without counting octave errors

% normal case
IsUsedEst = zeros(EstNoteNum, 1);                   % flag indicating if an estimated note has been matched
for gt_note = 1:GTNoteNum
    gt_onset = GTNote{gt_note}(1,1);                % ground-truth note onset time
    gt_offset = GTNote{gt_note}(1,end);             % ground-truth note offset time
    gt_freq = mean(GTNote{gt_note}(2,:));           % ground-truth note average frequency (in MIDI number)
    for est_note = 1:EstNoteNum
        if IsUsedEst(est_note) == 0
            est_onset = EstNote{est_note}(1,1);     % estimated note onset time
            est_offset = EstNote{est_note}(1,end);  % estimated note offset time
            est_freq = mean(EstNote{est_note}(2,:));% estimated note average frequency

            if bUseOffset == 0                      % offset difference is not involved in the criterion
                if abs(est_freq - gt_freq) <= 0.5 ...
                        && abs(est_onset - gt_onset) <= OnsetDiff
                    nCorrect = nCorrect + 1;        % match
                    AOR = AOR ...
                        + (min(est_offset, gt_offset) - max(est_onset, gt_onset)) ...
                        / (max(est_offset, gt_offset) - min(est_onset, gt_onset));
                    break;
                end
            else                                    % offset difference is involved in the criterion
                if abs(est_freq - gt_freq) <= 0.5 ...
                        && abs(est_onset - gt_onset) <= OnsetDiff ...
                        && abs(est_offset - gt_offset) <= OffsetDiff * (gt_offset-gt_onset)
                    nCorrect = nCorrect + 1;        % match
                    AOR = AOR ...
                        + (min(est_offset, gt_offset) - max(est_onset, gt_onset)) ...
                        / (max(est_offset, gt_offset) - min(est_onset, gt_onset));
                    break;
                end
            end
        end
    end
end

% chroma case
IsUsedEst = zeros(EstNoteNum, 1);
for gt_note = 1:GTNoteNum
    gt_onset = GTNote{gt_note}(1,1);                % ground-truth note onset time
    gt_offset = GTNote{gt_note}(1,end);             % ground-truth note offset time
    gt_freq = mod(mean(GTNote{gt_note}(2,:)), 12);  % ground-truth note average frequency (in MIDI number)
    for est_note = 1:EstNoteNum
        if IsUsedEst(est_note) == 0
            est_onset = EstNote{est_note}(1,1);     % estimated note onset time
            est_offset = EstNote{est_note}(1,end);  % estimated note offset time
            est_freq = mod(mean(EstNote{est_note}(2,:)), 12);   % estimated note average frequency

            if bUseOffset == 0                      % offset difference is not involved in the criterion
                if (mod(est_freq - gt_freq, 12) <= 0.5 || mod(gt_freq - est_freq, 12) <= 0.5) ...
                        && abs(est_onset - gt_onset) <= OnsetDiff
                    nChromaCorrect = nChromaCorrect + 1;    % match
                    ChromaAOR = ChromaAOR ...
                        + (min(est_offset, gt_offset) - max(est_onset, gt_onset)) ...
                        / (max(est_offset, gt_offset) - min(est_onset, gt_onset));
                    break;
                end
            else                                    % offset difference is involved in the criterion
                if (mod(est_freq - gt_freq, 12) <= 0.5 || mod(gt_freq - est_freq, 12) <= 0.5) ...
                        && abs(est_onset - gt_onset) <= OnsetDiff ...
                        && abs(est_offset - gt_offset) <= OffsetDiff * (gt_offset-gt_onset)
                    nChromaCorrect = nChromaCorrect + 1;    % match
                    ChromaAOR = ChromaAOR ...
                        + (min(est_offset, gt_offset) - max(est_onset, gt_onset)) ...
                        / (max(est_offset, gt_offset) - min(est_onset, gt_onset));
                    break;
                end
            end
        end
    end
end

% calculate statistics
Results_note.Pre = nCorrect/EstNoteNum;
Results_note.Rec = nCorrect/GTNoteNum;
Results_note.Fme = 2*Results_note.Pre*Results_note.Rec ...
    / (Results_note.Pre + Results_note.Rec);
Results_note.AOR = AOR/nCorrect;

Results_note.OctPre = nChromaCorrect/EstNoteNum;
Results_note.OctRec = nChromaCorrect/GTNoteNum;
Results_note.OctFme = 2*Results_note.OctPre*Results_note.OctRec ...
    / (Results_note.OctPre + Results_note.OctRec);
Results_note.OctAOR = ChromaAOR/nChromaCorrect;
