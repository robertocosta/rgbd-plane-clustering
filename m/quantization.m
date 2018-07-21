
function [q2,q] = quantization(im, from, to, quantizationLevels)
    im(isnan(im)) = 0;
    delta =(to-from)/quantizationLevels; 
    qVec = from:delta:to;
    qVec = qVec(2:end-1);
%     thresh = multithresh(im,glob.quantizationLevels);
%     thresh = min(im(:)):(max(im(:))-min(im(:)))/(glob.quantizationLevels):max(im(:));
%     thresh = thresh(2:end);
    q = imquantize(im,qVec);
%     middle = [qVec,to]-delta/2;
    middle = zeros(quantizationLevels,1);
    for i=1:length(middle)
        middle(i) = mean(im(q==i));
    end
    q2 = zeros(size(q));
    for i=1:size(q,1)
        for j=1:size(q,2)
            q2(i,j) = middle(q(i,j));
        end
    end    
end