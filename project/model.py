import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.nn import functional as F


class Multi_Head_Attention_LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, num_layers=1, num_classes=2, batch_size=1, num_heads=2):
        super(Multi_Head_Attention_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.layer2 = nn.Dropout(p=0.6)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Dropout(p=0.6)
        self.layer5 = nn.Linear(hidden_size, num_classes)
        self.num_heads = num_heads

        W = torch.zeros(num_heads, hidden_size).cuda()
        for i in range(0, num_heads):
            torch.manual_seed(114 + i)
            wi = torch.randn(hidden_size)
            W[i] = wi
        self.params = nn.Parameter(W)

    def forward(self, x, len_of_oneTr):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        batch_x_pack = rnn_utils.pack_padded_sequence(x, len_of_oneTr, batch_first=True).cuda()

        out, (h1, c1) = self.layer1(batch_x_pack, (h0, c0))
        outputs, lengths = rnn_utils.pad_packed_sequence(out, batch_first=True)

        permute_outputs = outputs.permute(1, 0, 2)  # word_size * batch_size * hidden_size
        # permute_outputs torch.Size([480, 64, 400])
        permute_permute_outputs = permute_outputs.permute(2, 1, 0)  # hidden_size * batch_size * word_size
        context_tensor = torch.zeros(self.num_heads, self.batch_size, 1, self.hidden_size).cuda()
        # [num_head,hidden_size,batch_size]
        h1 = h1.permute(1, 0, 2)  # 64*1*400
        for i in range(0, self.num_heads):
            wi = self.params[i]  # hidden_size
            wi = wi.unsqueeze(0)  # 1 * hidden_size
            wi = wi.unsqueeze(0)  # 1 * 1 * hidden_size
            tmp = wi * permute_outputs  # len * batch_size * hidden_size
            # ------------------
            '''
            atten_energies_i = torch.sum(h1*tmp, dim=2)     #len * batch_size
            atten_energies_i = atten_energies_i.t()         #batch_size * len
            scores_i = F.softmax(atten_energies_i, dim=1)   #
            scores_i = scores_i.unsqueeze(0)                #1 * batch_size * len
            context_tensor[i] = torch.sum(scores_i*permute_permute_outputs,dim=2)       #1 * batch_size * len
            '''
            # ------------------
            tmp = tmp.permute(1, 2, 0)  # 64*400*len

            atten_energies_i = torch.bmm(h1, tmp)  # 64*1*len
            scores_i = F.softmax(atten_energies_i, dim=2)  # 64*1*len
            tmp = tmp.permute(0, 2, 1)  # 64*len*400
            context_tensor[i] = torch.bmm(scores_i, tmp)  # 64*1*400       #hidden_size * batch_size

        mean_context_vector = torch.mean(context_tensor, dim=0)  # 64*400
        mean_context_vector = mean_context_vector.permute(1, 0, 2)
        h1 = h1.permute(1, 0, 2)
        out = h1 * mean_context_vector

        '''
        atten_energies = torch.sum(h1*permute_outputs, dim=2)
        #*表示逐元素相乘。[1, 64, 400]  * [480, 64, 400]  结果是torch.Size([480, 64, 400])
        #然后在最后一维上相加，就把最后一维去掉了。
        #对隐向量维求和
        #h1 torch.Size([1, 64, 400])
        #permute_outputs torch.Size([480, 64, 400])

        #atten_energies torch.Size([480, 64])

        atten_energies = atten_energies.t() #转置为torch.Size([64, 480])

        scores = F.softmax(atten_energies, dim=1) #表示输出h1,对之前每个h1i的注意力分值。

        scores = scores.unsqueeze(0)

        permute_permute_outputs = permute_outputs.permute(2,1,0)
        #permute_permute_outputs torch.Size([400, 64, 480])
        #scores torch.Size([1, 64, 480])   #permute_permute_outputs torch.Size([400, 64, 480])  
        # #逐元素相乘，要把后两个维度对其，然后有一个Tensor会自己复制扩展。结果就是  torch.Size([400, 64, 480])大小，然后用Sum将某一维消去即可实现点积的效果。
        context_vector = torch.sum(scores*permute_permute_outputs,dim=2)  #[400,64]  #用score作为权重对每一时间步的输出进行加权求和，得到h1对应的context向量。

        #再与h1拼接在一起，形成【1,64,800】的大小
        #h1 torch.Size([1, 64, 400])
        context_vector = context_vector.t()
        context_vector = context_vector.unsqueeze(0)
        #context_vector   torch.Size([1, 64, 400])
        out = torch.cat((h1,context_vector),2)
        #out torch.Size([1, 64, 800])
        '''
        out = self.layer2(out)  # dropout

        out = self.layer3(out)  # linear

        out = self.layer4(out)  # dropout

        out = self.layer5(out)  # linear

        return out  # torch.Size([1, 64, 85])
