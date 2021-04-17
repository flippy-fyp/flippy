function [FBD, FTD] = EvalScoFo_ASL(gtFile, resFile)
% Evalucate the score following results in ASL format
% In this format, the evaluation is done for each audio frame. Compare its
% aligned score position and its reference score position
%
% Input
%   - gtFile    : ground-truth alignment file
%   - resFile   : score following result file
% Output
%   - FBD       : frame-level beat difference
%   - FTD       : frame-level time difference
%
% Author: Zhiyao Duan
% Created: 9/17/2010
% Last modified: 9/17/2010

% read the ground-truth alignment file
fid = fopen(gtFile, 'r');
alignGT = fscanf(fid, '%d\t%d\t%f%d\n', [4 inf]);
fclose(fid);

% read the score following result file
fid = fopen(resFile, 'r');
alignRes = fscanf(fid, '%d\t%d\t%f%d\n', [4 inf]);
fclose(fid);

if size(alignGT, 2) ~= size(alignGT, 2)
    error('Size of ground-truth alignment and score following results does not match!');
end

% evaluate
idx = alignGT(3,:)~=0;                                      % do not consider those frames that don't have ground-truth aligned score positions
FBD = mean(abs(alignRes(3,idx)-alignGT(3,idx)));
FTD = mean(abs(alignRes(4,idx)-alignGT(4,idx)));
