from utils.cfg_parse import parse_cfg
from utils.custom_ops import EmptyLayer,DeformableConv2d
import torch
import torch.nn as nn
from torchsummary import summary

# print(model_list)


def create_blocks(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filter = 2
    output_filter = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if(x["type"] == "convolutional"):
            # print("0")
            activation = x['activation']
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x['pad'])
            kernel = int(x['size'])
            stride = int(x['stride'])


            conv = nn.Conv2d(prev_filter, filters, kernel,
                             stride, bias=bias)
            module.add_module("conv_{}".format(i), conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(i), bn)

            if activation == 'relu':
                activ_fn = nn.ReLU()
                module.add_module("ReLU{}".format(i),activ_fn)

            elif activation == "mish":
                activation_fn = nn.Mish()
                module.add_module("Mish_{}".format(i), activation_fn)
            
            elif activation == "leaky":
                activation_fn = nn.LeakyReLU()
                module.add_module("Leaky_ReLU{}".format(i),activation_fn)

        
        elif(x['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module("Shortcut_{}".format(i), shortcut)
        
        elif(x['type'] == 'pooling'):
            pooling = nn.MaxPool2d(kernel_size=2)
            module.add_module("Pooling2d_{}".format(i), pooling)
        
        elif(x['type'] == 'avgpool'):
            pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
            module.add_module("AvgPool{}".format(i),pool)
        
        elif(x['type'] == 'flatten'):
            fltn = nn.Flatten()
            module.add_module("Flatten{}".format(i),fltn)
        
        elif(x["type"] == "linear"):
            # print("0")
            #activation = x['activation']
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True

            filters = int(x["filters"])
    
            conv = nn.Linear(prev_filter, filters, bias=bias)
            module.add_module("linear_{}".format(i), conv)

            if batch_norm:
                bn = nn.BatchNorm1d(filters)
                module.add_module("batch_norm_{}".format(i), bn)

            if activation == 'relu':
                activ_fn = nn.ReLU()
                module.add_module("ReLU{}".format(i),activ_fn)

            elif activation == "mish":
                activation_fn = nn.Mish()
                module.add_module("Mish_{}".format(i), activation_fn)
            
            elif activation == "leaky":
                activation_fn = nn.LeakyReLU()
                module.add_module("Leaky_ReLU{}".format(i),activation_fn)

        
        
        '''elif(x['type'] == 'linear'):
            out_ftrs = x['filters']
            activ = x['activation']
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            lin = nn.Linear(512,out_ftrs)
            module.add_module("Linear{}".format(i),lin)

            if batch_norm:
                bn = nn.BatchNorm2d(out_ftrs)
                module.add_module("batch_norm_{}".format(i),bn)
            if activ == 'relu':
                activ_fn = nn.ReLU()
                module.add_module("ReLU{}".format(i),activ_fn)'''

        
        module_list.append(module)
        prev_filter = filters
        #print(prev_filter)
        output_filter.append(filters)
    return module_list


class CNN6(nn.Module):

    def __init__(self):
        super(CNN6, self).__init__()

        self.model_list = parse_cfg("cfgs/cnn14.cfg")
        self.module_list = create_blocks(self.model_list)
        #print(self.module_list)

        #self.fc1 = nn.Linear(512,2048)

    def forward(self, x):
        
        for i, module in enumerate(self.model_list[1:]):
            module_type = module['type']
            if module_type == "convolutional" or module_type=='flatten' or module_type=='linear' or module_type=="pooling" or module_type == "deformable" or module_type == "avgpool":
                x = self.module_list[i](x)

           
            

            
            #x = self.fc1(x)
            #print(x.shape)

        return x

model = CNN6()
t = torch.Tensor(4,2,128,3446)
tl = model(t)
#print(tl.shape)
