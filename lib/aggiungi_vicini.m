function ind = aggiungi_vicini(ind)
%AGGIUNGI_VICINI Summary of this function goes here
%   Detailed explanation goes here
mappa_vicinanza = const;
ind = mappa_vicinanza(ind,:);
ind = unique(ind(:));
ind(ind==0)=[];
end

