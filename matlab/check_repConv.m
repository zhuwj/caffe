clear,clc

net_model = '/data1/deep_action/models/google_net_singleGPU/deploy.prototxt';
net_weights = '/data1/deep_action/models/google_net_singleGPU/bvlc_googlenet.caffemodel';
net1 = caffe.Net(net_model, net_weights, 'train');

net_model2 = '/data1/deep_action/models/google_net_singleGPU/deploy_videodata.prototxt';
net_weights2 = '/data1/deep_action/models/google_net_singleGPU/bvlc_googlenet_fold_20.caffemodel';
net2 = caffe.Net(net_model2, net_weights2, 'train');

name_layer = 'conv1/7x7_s2';
layer1 = net1.layer_vec(net1.name2layer_index(name_layer));
param1 = layer1.params;
layer2 = net2.layer_vec(net2.name2layer_index(name_layer));
param2 = layer2.params;

for k = 1:length(layer1.params)
    blob1 = layer1.params(k).get_data(); %blob
    blob2 = layer2.params(k).get_data();
    
    if isvector(blob1) && isvector(blob2)
        assert(all(blob1 == blob2));
    else
        mblob = mean(blob1, 3);
        blob_rep = repmat(mblob, [1,1,20,1]);
        minmax(blob_rep(:)')
        minmax(blob2(:)')
        minmax(blob_rep(:)' - blob2(:)')
    end
end

