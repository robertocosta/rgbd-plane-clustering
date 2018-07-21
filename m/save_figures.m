function save_figures(figs, name, saveTitle)
    [~,~] = mkdir('img');
    for i=1:length(figs)
        set(0, 'currentfigure', figs{i});
        if nargin>2
            if ~saveTitle
                tit = matlab.graphics.primitive.Text;
                tit.String = '';
                set(gcf,Title,tit);
            end
        end
        saveas(figs{i},strcat('img/',figs{i}.Name,'_',name,'.png'));
    end
    
end

