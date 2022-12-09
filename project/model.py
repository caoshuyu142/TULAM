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
        self.layer3 = nn.Linear(2*hidden_size, hidden_size)
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
        permute_permute_outputs = permute_outputs.permute(2, 1, 0)  # hidden_size * batch_size * word_size
        context_tensor = torch.zeros(self.num_heads, self.batch_size, 1, self.hidden_size).cuda()
        # [num_head,hidden_size,batch_size]
        h1 = h1.permute(1, 0, 2)
        for i in range(0, self.num_heads):
            wi = self.params[i]
            wi = wi.unsqueeze(0) 
            wi = wi.unsqueeze(0) 
            tmp = wi * permute_outputs 
            tmp = tmp.permute(1, 2, 0) 

            atten_energies_i = torch.bmm(h1, tmp) 
            scores_i = F.softmax(atten_energies_i, dim=2)
            tmp = tmp.permute(0, 2, 1)
            context_tensor[i] = torch.bmm(scores_i, tmp)

        mean_context_vector = torch.mean(context_tensor, dim=0)
        mean_context_vector = mean_context_vector.permute(1, 0, 2)
        h1 = h1.permute(1, 0, 2)
        out = torch.cat((h1,mean_context_vector),2)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.layer5(out)

        return out
