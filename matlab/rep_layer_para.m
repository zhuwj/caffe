clear,clc

fold = 10 * 2;
net_model = '/data1/deep_action/models/bvlc_googlenet/deploy.prototxt';
net_weights_src = '/data1/deep_action/models/bvlc_googlenet/bvlc_googlenet.caffemodel';
net_weights_dst = ['/data1/deep_action/models/bvlc_googlenet/bvlc_googlenet_fold_', num2str(fold), '.caffemodel'];
net = caffe.Net(net_model, net_weights_src, 'train');

% name = 'conv1/7x7_s2';
% id = net.name2layer_index(name);
% 
% %revise blobs
% params = net.layer_vec(id).params;
% for n = 1:length(params)
%     data_old = params(n).get_data();
%     shape_old = size(data_old);    
%     if length(shape_old) == 4
%         shape_old,
%         shape_new = shape_old; shape_new(3) = fold;
%         %mean across channels
%         data_old = mean(data_old, 3);
%         
%         %repeat channels
%         data_new = repmat(data_old, [1,1,fold, 1]);
%         assert(all(size(data_new) == shape_new));
%         params(n).reshape(shape_new)
%         params(n).set_data(data_new);
%     elseif length(shape_old) ~= 2
%         error('blob should be 4D data or vector');
%     end
% end
% net.layer_vec(id).params = params;

%save new model
net.save(net_weights_dst);