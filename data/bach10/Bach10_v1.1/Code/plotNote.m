function plotNote(Note, track)
% Plot notes in some track
% Input
%   - Note      : Notes of all the tracks
%   - track     : Track label
% Author: Zhiyao Duan
% Created: 5/5/2009
% Last modified: 7/22/2009

NoteNum = length(Note{track});
for currN = 1:NoteNum
    hold on;
    switch track
        case 1
            plot(Note{track}{currN}(1,:), Note{track}{currN}(2,:), 'b');
        case 2
            plot(Note{track}{currN}(1,:), Note{track}{currN}(2,:), 'g');
        case 3
            plot(Note{track}{currN}(1,:), Note{track}{currN}(2,:), 'r');
        case 4
            plot(Note{track}{currN}(1,:), Note{track}{currN}(2,:), 'c');
        otherwise
            plot(Note{track}{currN}(1,:), Note{track}{currN}(2,:), 'k');
    end
end
