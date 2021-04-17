% Generate all combinations of different parts of each piece
% This will give us 1 solo, 6 duets, 4 trios and 1 quartet for each piece
%
% Author: Zhiyao Duan
% Created: 9/10/2010
% Last modified: 1/15/2012

clc; clear;

% generate the following files or not
bSolo = 0;
bDuet = 0;
bTrio = 0;
bQuartet = 1;                                                           % this is the same as original quartets, just to make naming convention complete

wavPath = 'D:\Work\My_data\Bach10\';                                    % the folder containing testing music pieces
wavFileList = {
    '01-AchGottundHerr';
    '02-AchLiebenChristen';
    '03-ChristederdubistTagundLicht';
    '04-ChristeDuBeistand';
    '05-DieNacht';
    '06-DieSonne';
    '07-HerrGott';
    '08-FuerDeinenThron';
    '09-Jesus';
    '10-NunBitten'
    };                                                                  % test file lists, each one is a folder
wavNum = size(wavFileList, 1);

for fnum = 1:wavNum
    % read midi
    midMixFile = strcat(wavPath, wavFileList{fnum}, '\', wavFileList{fnum}, '.mid');
    nmatMix = readmidi_java(midMixFile);
    % read audio mixture
    wavMixFile = strcat(wavPath, wavFileList{fnum}, '\', wavFileList{fnum}, '.wav');
    [wavdataMix, fs] = wavread(wavMixFile);
    % read audio sources
    wavSrcFile = cell(4,1);
    wavSrcFile{1} = strrep(midMixFile, '.mid', '-violin.wav');
    wavSrcFile{2} = strrep(midMixFile, '.mid', '-clarinet.wav');
    wavSrcFile{3} = strrep(midMixFile, '.mid', '-saxphone.wav');
    wavSrcFile{4} = strrep(midMixFile, '.mid', '-bassoon.wav');
    wavdata = cell(4,1);
    [wavdata{1}] = wavread(wavSrcFile{1});
    [wavdata{2}] = wavread(wavSrcFile{2});
    [wavdata{3}] = wavread(wavSrcFile{3});
    [wavdata{4}] = wavread(wavSrcFile{4});
    % read MIREX ground-truth alignment file
    fid = fopen(strrep(wavMixFile, '.wav', '.txt'), 'r');
    alignGT = fscanf(fid, '%d\t%d\t%d\t%d\r\n', [4 inf]);
    alignGT = alignGT';
    fclose(fid);
    % read GTF0s
    load(strrep(wavMixFile, '.wav', '-GTF0s.mat'));
    origGTF0s = GTF0s;
    % read GTNotes
    load(strrep(wavMixFile, '.wav', '-GTNotes.mat'));
    origGTNotes = GTNotes;

    if bSolo == 1
        % generate solos
        genNum = 0;
        for i = 1:4
            genNum = genNum + 1;
            % create folder
            genPath = strcat(wavPath, sprintf('BachChorale-1-%02d-%d', fnum, genNum));
            if exist(genPath,'dir')==7
                rmdir(genPath,'s');                                     % remove old files
            end
            mkdir(genPath);
            % generate files
            genFile = strcat(genPath, '\', sprintf('BachChorale-1-%02d-%d', fnum, genNum));
            % generate the mixture wave file
            wavmix = wavdata{i};
            wavmix = 0.95*wavmix/max(abs(wavmix));
            wavwrite(wavmix, fs, strcat(genFile, '.wav'));
            % generate the MIDI file
            nmatNewMix = nmatMix(nmatMix(:,3)==i, :);
            nmatTempo = 60*(max(nmatMix(:,1))-min(nmatMix(:,1)))/(max(nmatMix(:,6))-min(nmatMix(:,6)));
            writemidi_java(nmatNewMix, strcat(genFile, '.mid'), 120, nmatTempo);
            % generate MIREX alignment file
            fid = fopen(strcat(genFile, '.txt'), 'w');
            idx = alignGT(:,4)==i;
            tmpAlignGT = alignGT(idx, :);
            for k = 1:size(tmpAlignGT, 1)
                fprintf(fid, '%d\t%d\t%d\t%d\r\n', tmpAlignGT(k,1), tmpAlignGT(k,2), tmpAlignGT(k,3), tmpAlignGT(k,4));
            end
            fclose(fid);
            % generate ASL alignment file
            copyfile(strrep(wavMixFile, '.wav', '.asl'), strcat(genFile, '.asl'));
            % generate GTF0s file
            GTF0s = origGTF0s(i,:);
            save(strcat(genFile, '-GTF0s.mat'), 'GTF0s');
            % generate GTNotes file
            GTNotes = origGTNotes(i);
            save(strcat(genFile, '-GTNotes.mat'), 'GTNotes');
        end
    end

    if bDuet == 1
        % generate duets
        genNum = 0;
        for i = 1:3                                                         % make sure no repeats
            for j = i+1:4
                genNum = genNum + 1;
                % create folder
                genPath = strcat(wavPath, sprintf('BachChorale-2-%02d-%d', fnum, genNum));
                if exist(genPath,'dir')==7
                    rmdir(genPath,'s');                                     % remove old files
                end
                mkdir(genPath);
                % generate files
                genFile = strcat(genPath, '\', sprintf('BachChorale-2-%02d-%d', fnum, genNum));
                % generate the mixture wave file
                wavmix = wavdata{i} + wavdata{j};
                wavmix = 0.95*wavmix/max(abs(wavmix));
                wavwrite(wavmix, fs, strcat(genFile, '.wav'));
                % generate the source wave files
                copyfile(wavSrcFile{i}, strcat(genFile, sprintf('-1-%d.wav', i)));
                copyfile(wavSrcFile{j}, strcat(genFile, sprintf('-2-%d.wav', j)));
                % generate the MIDI file
                nmatNewMix = nmatMix(nmatMix(:,3)==i|nmatMix(:,3)==j, :);
                nmatTempo = 60*(max(nmatMix(:,1))-min(nmatMix(:,1)))/(max(nmatMix(:,6))-min(nmatMix(:,6)));
                writemidi_java(nmatNewMix, strcat(genFile, '.mid'), 120, nmatTempo);
                % generate MIREX alignment file
                fid = fopen(strcat(genFile, '.txt'), 'w');
                idx = (alignGT(:,4)==i | alignGT(:,4)==j) ;
                tmpAlignGT = alignGT(idx, :);
                for k = 1:size(tmpAlignGT, 1)
                    fprintf(fid, '%d\t%d\t%d\t%d\r\n', tmpAlignGT(k,1), tmpAlignGT(k,2), tmpAlignGT(k,3), tmpAlignGT(k,4));
                end
                fclose(fid);
                % generate ASL alignment file
                copyfile(strrep(wavMixFile, '.wav', '.asl'), strcat(genFile, '.asl'));
                % generate GTF0s file
                GTF0s = origGTF0s([i;j],:);
                save(strcat(genFile, '-GTF0s.mat'), 'GTF0s');
                % generate GTNotes file
                GTNotes = origGTNotes([i;j]);
                save(strcat(genFile, '-GTNotes.mat'), 'GTNotes');
            end
        end
    end

    if bTrio == 1
        % generate trios
        genNum = 0;
        for i = 1:4
            genNum = genNum + 1;
            % being used sources
            idx = 1:4;
            idx = idx(idx~=i);
            % create folder
            genPath = strcat(wavPath, sprintf('BachChorale-3-%02d-%d', fnum, genNum));
            if exist(genPath,'dir')==7
                rmdir(genPath,'s');                                     % remove old files
            end
            mkdir(genPath);
            % generate files
            genFile = strcat(genPath, '\', sprintf('BachChorale-3-%02d-%d', fnum, genNum));
            % generate the mixture wave file
            wavmix = wavdata{idx(1)} + wavdata{idx(2)} + wavdata{idx(3)};
            wavmix = 0.95*wavmix/max(abs(wavmix));
            wavwrite(wavmix, fs, strcat(genFile, '.wav'));
            % generate the source wave files
            for j = 1:3
                copyfile(wavSrcFile{idx(j)}, strcat(genFile, sprintf('-%d-%d.wav', j, idx(j))));
            end
            % generate the MIDI file
            nmatNewMix = nmatMix(nmatMix(:,3)~=i, :);
            nmatTempo = 60*(max(nmatMix(:,1))-min(nmatMix(:,1)))/(max(nmatMix(:,6))-min(nmatMix(:,6)));
            writemidi_java(nmatNewMix, strcat(genFile, '.mid'), 120, nmatTempo);
            % generate MIREX alignment file
            fid = fopen(strcat(genFile, '.txt'), 'w');
            idx = alignGT(:,4)~=i;
            tmpAlignGT = alignGT(idx, :);
            for k = 1:size(tmpAlignGT, 1)
                fprintf(fid, '%d\t%d\t%d\t%d\r\n', tmpAlignGT(k,1), tmpAlignGT(k,2), tmpAlignGT(k,3), tmpAlignGT(k,4));
            end
            fclose(fid);
            % generate ASL alignment file
            copyfile(strrep(wavMixFile, '.wav', '.asl'), strcat(genFile, '.asl'));
            % generate GTF0s file
            idx = ([1;2;3;4]~=i);
            GTF0s = origGTF0s(idx,:);
            save(strcat(genFile, '-GTF0s.mat'), 'GTF0s');
            % generate GTNotes file
            GTNotes = origGTNotes(idx);
            save(strcat(genFile, '-GTNotes.mat'), 'GTNotes');
        end
    end

    if bQuartet == 1                                                % just copy original files
        % generate quartets
        genNum = 1;
        % create folder
        genPath = strcat(wavPath, sprintf('BachChorale-4-%02d-%d', fnum, genNum));
        if exist(genPath,'dir')==7
            rmdir(genPath,'s');                                     % remove old files
        end
        mkdir(genPath);
        % generate files
        genFile = strcat(genPath, '\', sprintf('BachChorale-4-%02d-%d', fnum, genNum));
        % generate the mixture wave file
        copyfile(wavMixFile, strcat(genFile, '.wav'));
        % generate the source wave files
        for j = 1:4
            copyfile(wavSrcFile{j}, strcat(genFile, sprintf('-%d-%d.wav', j, j)));
        end
        % generate the MIDI file
        copyfile(midMixFile, strcat(genFile, '.mid'));
        % generate MIREX alignment file
        copyfile(strrep(wavMixFile, '.wav', '.txt'), strcat(genFile, '.txt'));
        % generate ASL alignment file
        copyfile(strrep(wavMixFile, '.wav', '.asl'), strcat(genFile, '.asl'));
        % generate GTF0s file
        copyfile(strrep(wavMixFile, '.wav', '-GTF0s.mat'), strcat(genFile, '-GTF0s.mat'));
        % generate GTNotes file
        copyfile(strrep(wavMixFile, '.wav', '-GTNotes.mat'), strcat(genFile, '-GTNotes.mat'));
    end
end
