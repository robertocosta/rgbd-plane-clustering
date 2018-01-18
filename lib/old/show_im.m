function f = show_im(frame,depth,label)
%SHOW_IM Summary of this function goes here
%   Detailed explanation goes here
f = figure;
subplot(2,2,1);
imshow(frame);
subplot(2,2,2);
imshow(depth/max(max(depth)));
subplot(2,2,3);
im_labeled = zeros(size(frame,1),size(frame,2),3);
n = max(max(label))+1;
rng('default');
colors = rand(n,3);
for i=1:size(frame,1)
    for j=1:size(frame,2)
        im_labeled(i,j,:)=colors(label(i,j)+1,:);
    end
end
imshow(im_labeled);
end

