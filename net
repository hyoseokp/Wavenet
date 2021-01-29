class Wavenet(nn.Module):
    def __init__(self):
        super(Wavenet,self).__init__()
        self.input_conv = nn.Conv1d(in_channels=1,
                                    out_channels=128,
                                    kernel_size=1)
        self.dilated_conv1d_1_1 = nn.ModuleList()
        self.dilated_conv1d_2_1 = nn.ModuleList()
        self.conv1x1_res_1 = nn.ModuleList()
        self.conv1x1_skip_1 = nn.ModuleList()
        
        self.dilated_conv1d_1_2 = nn.ModuleList()
        self.dilated_conv1d_2_2 = nn.ModuleList()
        self.conv1x1_res_2 = nn.ModuleList()
        self.conv1x1_skip_2 = nn.ModuleList()

        self.dilated_conv1d_1_3 = nn.ModuleList()
        self.dilated_conv1d_2_3 = nn.ModuleList()
        self.conv1x1_res_3 = nn.ModuleList()
        self.conv1x1_skip_3 = nn.ModuleList()
        
        self.dilations_1 = []
        self.dilations_2 = []
        self.dilations_3 = []
#         self.pad = []
        number_of_layer = 10  #the number of dilation layer * the number of stack
        
        for i in range(number_of_layer): # 2**number_of_layer = dilated num
            dilation = 2**i
            self.dilations_1.append(dilation)
#             self.pad.append(nn.ConstantPad1d((dilation, dilation), 0))
            
            self.dilated_conv1d_1_1.append(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   dilation=dilation))
            self.dilated_conv1d_2_1.append(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   dilation=dilation))
            self.conv1x1_res_1.append(nn.Conv1d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=1,
                                              dilation=1))
            self.conv1x1_skip_1.append(nn.Conv1d(in_channels=128,
                                               out_channels=128,
                                               kernel_size=1,
                                               dilation=1))
            
        for i in range(number_of_layer): # 2**number_of_layer = dilated num
            dilation = 2**i
            self.dilations_2.append(dilation)
#             self.pad.append(nn.ConstantPad1d((dilation, dilation), 0))
            
            self.dilated_conv1d_1_2.append(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   dilation=dilation))
            self.dilated_conv1d_2_2.append(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   dilation=dilation))
            self.conv1x1_res_2.append(nn.Conv1d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=1,
                                              dilation=1))
            self.conv1x1_skip_2.append(nn.Conv1d(in_channels=128,
                                               out_channels=128,
                                               kernel_size=1,
                                               dilation=1))
        for i in range(number_of_layer): # 2**number_of_layer = dilated num
            dilation = 2**i
            self.dilations_3.append(dilation)
#             self.pad.append(nn.ConstantPad1d((dilation, dilation), 0))
            
            self.dilated_conv1d_1_3.append(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   dilation=dilation))
            self.dilated_conv1d_2_3.append(nn.Conv1d(in_channels=128,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   dilation=dilation))
            self.conv1x1_res_3.append(nn.Conv1d(in_channels=128,
                                              out_channels=128,
                                              kernel_size=1,
                                              dilation=1))
            self.conv1x1_skip_3.append(nn.Conv1d(in_channels=128,
                                               out_channels=128,
                                               kernel_size=1,
                                               dilation=1))
        self.output_conv1 = nn.Conv1d(in_channels=128,
                                      out_channels=2048,
                                      kernel_size=3)
        
        self.output_conv2 = nn.Conv1d(in_channels=2048,
                                      out_channels=256,
                                      kernel_size=3)
        self.output_conv_final = nn.Conv1d(in_channels=256,
                                           out_channels=1,
                                           kernel_size=1)
        self.relu = nn.ReLU()
#         self.max_dilation = max(self.dilations)
#         self.device = 'cuda'
#         self.to(device)

    def forward(self,x):
        x = self.input_conv(x)
        skip = torch.zeros((x.shape[0], x.shape[1], x.shape[2]),
                           dtype=torch.float).cuda()
#         skip = torch.zeros((x.shape[0], x.shape[1], x.shape[2]),
#                            dtype=torch.float)

        for i, dilation in enumerate(self.dilations_1):
            
            fx1 = self.dilated_conv1d_1_1[i](x)
            fx2 = torch.tanh(fx1)
            gx1 = self.dilated_conv1d_2_1[i](x)
            gx2 = torch.sigmoid(gx1)
            hx  = torch.mul(fx2,gx2)
            
            skip = skip[:,:,dilation:-dilation]
            skip = skip.clone() + self.conv1x1_skip_1[i](hx)
            
            x = x[:,:,dilation:-dilation]
            x = x.clone() + self.conv1x1_res_1[i](hx)
            
        # stack 2
        
        for i, dilation in enumerate(self.dilations_2):
            
            fx1 = self.dilated_conv1d_1_2[i](x)
            fx2 = torch.tanh(fx1)
            gx1 = self.dilated_conv1d_2_2[i](x)
            gx2 = torch.sigmoid(gx1)
            hx  = torch.mul(fx2,gx2)
            
            skip = skip[:,:,dilation:-dilation]
            skip = skip.clone() + self.conv1x1_skip_2[i](hx)
            
            x = x[:,:,dilation:-dilation]
            x = x.clone() + self.conv1x1_res_2[i](hx)    
            
        # stack 3
        
        for i, dilation in enumerate(self.dilations_3):
            
            fx1 = self.dilated_conv1d_1_3[i](x)
            fx2 = torch.tanh(fx1)
            gx1 = self.dilated_conv1d_2_3[i](x)
            gx2 = torch.sigmoid(gx1)
            hx  = torch.mul(fx2,gx2)

            skip = skip[:,:,dilation:-dilation]
            skip = skip.clone() + self.conv1x1_skip_3[i](hx)
            
            x = x[:,:,dilation:-dilation]
            x = x.clone() + self.conv1x1_res_3[i](hx)
            
        y = self.relu(skip)
#         print(y.size())
        y = self.output_conv1(nn.ConstantPad1d((1, 1), 0)(y))
        y = self.relu(y)
        x = self.output_conv2(nn.ConstantPad1d((1, 1), 0)(y))
        x = self.output_conv_final(x)
        return x

    





