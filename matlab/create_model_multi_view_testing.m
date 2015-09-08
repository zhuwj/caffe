deploy_src = '/data1/tools/caffe/models/bvlc_reference_caffenet/deploy.prototxt';
model_src = '/data1/tools/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';

deploy_dst = '/data1/tools/caffe/models/bvlc_reference_caffenet/deploy_conv.prototxt';
model_dst = '/data1/tools/caffe/models/bvlc_reference_caffenet/convmodel.caffemodel';

net1 = caffe.Net(deploy_src, model_src, 'test');
net2 = caffe.Net(deploy_dst, model_dst, 'test');
% net.save(path_model_dst);

assert(length(net1.layer_vec) == length(net2.layer_vec));

for k = 1:length(net1.layer_vec)
    name1 = net1.layer_names{k};
    name2 = net2.layer_names{k};
    
    if ~isequal(name1, name2)
        assert(isequal(name1, name2(1:strfind(name2, '_') - 1)));
    end
    
    for n = 1:length(net1.layer_vec(k).params)
        A = net1.layer_vec(k).params(1).get_data();
        B = net2.layer_vec(k).params(1).get_data();
        assert(isequal(A(:), B(:)));
    end
end