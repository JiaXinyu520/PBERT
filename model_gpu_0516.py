#%%
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import _pickle as pk
import sys
import os
import time
from random import *

def is_valid_seq(seq, max_len=2000):
	l = len(seq)
	valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
	if (l <= max_len) and set(seq) <= set(valid_aas):
		return True
	else:
		return False

def seqloader(filename,maxseqlen=300):
        pkl_input = open(filename, 'rb')
        raw = pk.load(pkl_input, encoding='bytes')
        pkl_input.close()
        sentences=[]
        for t in raw:
                s=t[b'sequence'].decode()
                if( len(s) <= 300 ):
                        sentences.append([s])
        return sentences

sentences = seqloader('/export/home/prpstudent/jxy/transformer/dataset/pdb25-6767-train.release.contactFeatures.pkl')  
e_sentences = seqloader('/export/home/prpstudent/jxy/transformer/dataset/pdb25-6767-valid.release.contactFeatures.pkl')

#%%

word2idx = {'[pad]':0, '[mask]':1,'[start]':2, 'A' : 3, 'R' : 4, 'N' : 5, 'D' : 6, 'B' : 7, 'C' : 8, 'Q' : 9, 'E' : 10, 'Z': 11, 'G': 12, 'H':13, 'I':14, 'L':15, 'K':16, 'M':17, 'F' :18 , 'P':19, 'S':20 ,'T':21, 'W':22, 'Y':23, 'V':24, '[end]':25 }
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)
batch_size = 256
e_batch_size = 32
max_len = 300 # enc_input max sequence length
max_pred = 10
n_layers = 6
n_heads = 8
d_model = 64
d_ff = d_model*4
d_k = d_v = 32


def make_data(sentences):
        sentences_in=sentences[:]
        batch=[]
        for i in range(len(sentences_in)):
                inter = []
                inter.append('[start]')
                for j in sentences_in[i][0]:
                        inter.append(j)
                inter.append('[end]')
                sentences_in[i]=inter[:]
                n_real = min(300,len(inter))
                
                if len(sentences_in[i])<=max_len:
                        n = max_len - len(sentences_in[i])
                        pad_list = ['[pad]' for j in range(n)]
                        sentences_in[i].extend(pad_list)
                else:
                        sentences_in[i] = sentences_in[i][:max_len]
                enc_input = [word2idx[n] for n in sentences_in[i]]

                n_pred =  min(max_pred, max(1, int(n_real * 0.10)))
                cand_masked_pos = [ k for k in range(n_real)]

                shuffle(cand_masked_pos)
                masked_token, masked_pos= [], []
                for pos in cand_masked_pos[:max_pred]:
                        masked_pos.append(pos)
                        masked_token.append(enc_input[pos])
                        randnum = random()
                        if randnum < 0.8:
                                enc_input[pos] = word2idx['[mask]']
                        elif randnum > 0.9:
                                enc_input[pos] = randint(2, vocab_size-1)
                if len(masked_pos) < max_pred:
                        num_start = max_pred - len(pos)
                        masked_pos.extend([0] * num_start)
                        masked_token.extend(['[start]'] * num_start)

                batch.extend([[enc_input,masked_token, masked_pos]])

        return batch
batch = make_data(sentences)
input_ids, masked_tokens, masked_pos = zip(*batch)
input_ids, masked_tokens, masked_pos = torch.LongTensor(input_ids),\
        torch.LongTensor(masked_tokens),torch.LongTensor(masked_pos) #input_ids:[seq_num,seq_len],masked_tokens,masked_pos:[seq_num,seq_len]
e_batch = make_data(e_sentences)
e_input_ids, e_masked_tokens, e_masked_pos = zip(*e_batch)
e_input_ids, e_masked_tokens, e_masked_pos = torch.LongTensor(e_input_ids),\
        torch.LongTensor(e_masked_tokens),torch.LongTensor(e_masked_pos) #input_ids:[seq_num,seq_len],masked_tokens,masked_pos:[seq_num,seq_len]

# %%
class MyDataSet(Data.Dataset):
        def __init__(self, input_ids, masked_pos, masked_token):
                super(MyDataSet, self).__init__()
                self.input_ids = input_ids
                self.masked_pos = masked_pos
                self.masked_token = masked_token
        
        def __len__(self):
                return self.input_ids.shape[0]
        
        def __getitem__(self, idx):
                return self.input_ids[idx], self.masked_pos[idx], self.masked_token[idx]

loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos ), batch_size, True)  # batch_size   , shuffle = True
e_loader = Data.DataLoader(MyDataSet(e_input_ids, e_masked_tokens, e_masked_pos ), e_batch_size, True)  # batch_size   , shuffle = True

#%% 
def get_attn_pad_mask(seq_q, seq_k):
        batch_size, seq_len =seq_q.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_q.data.eq(0).unsqueeze(1) #[batch_size, 1, seq_len]
        return pad_attn_mask.expand(batch_size, seq_len, seq_len) #[batch_size, seq_len, seq_len]

def gelu(x):
        '''
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        '''
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
        def __init__(self):
                super(Embedding, self).__init__()
                self.tok_embed = nn.Embedding(vocab_size, d_model)
                self.pos_embed = nn.Embedding(max_len, d_model)
                self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
                seq_len = x.size(1)
                pos = torch.arange(seq_len, dtype=torch.long).to(device)
                pos = pos.unsqueeze(0).expand_as(x)
                embedding = self.tok_embed(x) + self.pos_embed(pos)
                return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
        def __init__(self):
                super(ScaledDotProductAttention,self).__init__()
        
        def forward(self, Q, K, V, attn_mask):
                scores = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(d_k)
                scores.masked_fill(attn_mask, -1e9)# 
                attn = nn.Softmax(dim=-1)(scores)
                context = torch.matmul(attn, V)
                return context

class MultiHeadAttention(nn.Module):
        def __init__(self):
                super(MultiHeadAttention, self).__init__()
                self.W_Q = nn.Linear(d_model, d_k * n_heads)
                self.W_K = nn.Linear(d_model, d_k * n_heads)
                self.W_V = nn.Linear(d_model, d_v * n_heads)
                self.Outlinear= nn.Linear(n_heads * d_v, d_model)
                self.LayerNorm = nn.LayerNorm(d_model) 

        def forward(self, Q, K, V, attn_mask):
                # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
                residual, batch_size = Q, Q.size(0)
                # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
                q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
                k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
                v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

                attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask

                # context:[batch_size, n_heads, seq_len, d_v], attn:[batch_size, n_heads, seq_len, seq_len]
                context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
                context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
                output = self.Outlinear(context)
                output = self.LayerNorm(output + residual)
                return output # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
        def __init__(self):
                super(PoswiseFeedForwardNet, self).__init__()
                self.fc1 = nn.Linear(d_model, d_ff)
                self.fc2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
                # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
                out = self.fc1(x)
                out = gelu(out)
                out = self.fc2(out)
                return out

class EncoderLayer(nn.Module):
        def __init__(self):
                super(EncoderLayer, self).__init__()
                self.enc_self_attn = MultiHeadAttention()
                self.pos_ffn = PoswiseFeedForwardNet()
        
        def forward(self, enc_inputs, enc_self_attn_mask):
                enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
                enc_outputs = self.pos_ffn(enc_outputs)
                return enc_outputs

class BERT(nn.Module):
        def __init__(self):
                super(BERT, self).__init__()
                self.embedding = Embedding()
                self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
                self.fc = nn.Sequential(
                        nn.Linear(d_model,d_model),
                        nn.Dropout(0.5),
                        nn.Tanh(),
                )
                self.classifier = nn.Linear(d_model, 2)
                self.linear = nn.Linear(d_model, d_model)
                self.active2 = gelu
                # fc2 is shared with embedding layer
                embed_weight = self.embedding.tok_embed.weight
                self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
                self.fc2.weight = embed_weight
        
        def forward(self, input_ids, masked_pos):
                output = self.embedding(input_ids) #[batch_size, maxlen, maxlen]
                enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) #[batch_size, maxlen, maxlen]
                for layer in self.layers:
                        #output:[batch_size, max_len, d_model]
                        output = layer(output, enc_self_attn_mask)
                masked_pos = masked_pos[:,:,None].expand(-1,-1,d_model)#[batch_size, max_pred, d_model]
                h_masked = torch.gather(output, 1, masked_pos) #masking position [batch_size, max_pred, d_model]
                h_masked = self.active2(self.linear(h_masked))
                logits_lm = self.fc2(h_masked)
                return logits_lm


def evaluate_accuracy(net,testdata,device):
        with torch.no_grad():
                loss_sum = 0
                num_i = 0
                acc_num=0
                pre_num=0
                for input_ids,masked_tokens,masked_pos in testdata:
                        net.eval()
                        num_i += 1
                        input_ids = input_ids.to(device)
                        masked_tokens = masked_tokens.to(device)
                        masked_pos = masked_pos.to(device)
                        logits_lm = model(input_ids, masked_pos)
                        y_hat=logits_lm.view(-1,vocab_size).argmax(1)
                        y_real = masked_tokens.view(-1)
                        acc_num += (y_hat == y_real).sum()
                        pre_num += len(y_hat)
                        loss = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
                        loss = (loss.float()).mean()
                        loss_sum +=loss
        return acc_num/ pre_num, loss_sum/num_i

def bestmodel(model, acc , eva_acc, max_acc, max_eva):
        if acc > max_acc:
                PATH1 = pro_filename + 'experiment/'+exp_name+'best/train_model_'
                PATH2 = pro_filename + 'experiment/'+exp_name+'best/train_model_dict'
                torch.save(model, PATH1) 
                torch.save(model.state_dict(),PATH2)
                max_acc = acc

        if eva_acc > max_eva:
                PATH1 = pro_filename + 'experiment/'+exp_name+'best/eval_model_'
                PATH2 = pro_filename + 'experiment/'+exp_name+'best/eval_model_dict'
                torch.save(model, PATH1) 
                torch.save(model.state_dict(),PATH2)
                max_eva = eva_acc
        
        return max_acc ,max_eva


#%% train
device = torch.device("cpu")
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = BERT()
model.to(device)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = optim.Adadelta(model.parameters(),lr=0.001)

pro_filename = '/export/home/prpstudent/jxy/transformer/'
exp_name = 'exp_1_1/'

max_acc = 0
max_e_acc = 0


for epoch in range(300):
        model.train()
        if epoch > 0:
                acc = acc_num/pre_num
                print('Epoch:', epoch, 'loss =', '{:.4f}'.format(loss_sum/num_i), 'train accuracy = {:.2%}'.format(acc.item()))
                e_acc, e_loss = evaluate_accuracy(model, e_loader, device)
                print('Evaluate loss = ', '{:.4f}'.format(e_loss),'evaluate accuracy = {:.2%}'.format(e_acc.item()) )
                max_acc, max_eva = bestmodel(model, acc ,e_acc, max_acc,max_e_acc)
                usetime = time.time() - start_time
                print('time' ,usetime)

        if epoch % 10 == 0 and epoch >0 :
                PATH1 = pro_filename + 'experiment/'+exp_name+'model/model_'+str(epoch)
                PATH2 = pro_filename + 'experiment/'+exp_name+'model/model_dict'+str(epoch)
                torch.save(model, PATH1) 
                torch.save(model.state_dict(),PATH2)
                print('so far epoch = ',epoch,' best_eva =',max_eva.item(),'best_acc =',max_acc.item(),'\n')
                

        loss_sum = 0
        num_i = 0
        acc_num=0
        pre_num=0
        start_time = time.time()
        for input_ids, masked_tokens, masked_pos in loader:
                num_i += 1
                input_ids = input_ids.to(device)
                masked_tokens = masked_tokens.to(device)
                masked_pos = masked_pos.to(device)
                logits_lm = model(input_ids, masked_pos)

                y_hat=logits_lm.view(-1,vocab_size).argmax(1)
                y_real = masked_tokens.view(-1)
                acc_num += (y_hat == y_real).sum()
                pre_num += len(y_hat)

                loss = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
                loss = (loss.float()).mean()
                loss_sum +=loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                


        