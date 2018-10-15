import torch
import torch.nn as nn
from torch.autograd import Variable


class Seq2seq(nn.Module):
    def __init__(self, emb_dim, rnn_hidden, num_layers, src_vocab_size, tar_vocab_size, bi, attn, attn_type, attn_dim):
        super(Seq2seq, self).__init__()

        self.emb_dim = emb_dim
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.bi = bi
        self.attn = attn
        self.attn_type = attn_type
        self.attn_dim = attn_dim

        self.src_emb = nn.Embedding(src_vocab_size, self.emb_dim)
        self.tar_emb = nn.Embedding(tar_vocab_size, self.emb_dim)
        self.encoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            bidirectional=bi,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            bidirectional=bi,
            batch_first=True
        )
        rnn_dim = rnn_hidden * 2 if bi else rnn_hidden
        if attn:
            self.attention = Attention(rnn_dim, self.attn_dim, self.attn_type)

        self.out_proj = nn.Linear(rnn_dim, tar_vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax()
        self.log_sm = nn.LogSoftmax(-1)

    # TODO why this stupid model keeps generating <SOS>? NEED ANY HELP, YOU XXX?
    def forward(self, src, tar):
        src_emb = self.src_emb(src)

        encoded, hidden = self.encoder(src_emb)

        tar_emb = self.tar_emb(tar)
        decoded, _ = self.decoder(tar_emb, hidden)

        score = None
        if self.attn:
            decoded, score = self.attention(encoded, decoded)

        out = self.out_proj(decoded)

        # out = self.sigmoid(out)
        out = self.log_sm(out)

        return out, score


    def translate(self, src, maxlen=50, SOS=0, EOS=1):
        # Greedy translate
        batch, src_seq = src.shape
        src_emb = self.src_emb(src)

        encoded, hidden = self.encoder(src_emb)

        translations = []
        attentions = []
        for b in range(batch):
            trans = []
            attn = []
            tar = torch.ones(1, 1, dtype=torch.long) * SOS
            cur_enc = encoded[b, :, :].unsqueeze(0)
            cur_hidden = tuple([h[:, 0, :].unsqueeze(1).contiguous() for h in hidden])
            for i in range(maxlen):
                pred, cur_hidden, one_attn = self.translate_one(cur_enc, tar.cuda(), cur_hidden)

                if pred == EOS:
                    break

                trans.append(pred.item())
                attn.append(one_attn)
                tar = pred

            translations.append(trans)
            attentions.append(attn)

        return translations, attentions

    def translate_one(self, enc, tar_inp_idx, hidden):
        tar_inp = self.tar_emb(tar_inp_idx)

        decoded, hidden = self.decoder(tar_inp, hidden)

        attn = None
        if self.attn:
            decoded, attn = self.attention(enc, decoded)

        out = self.out_proj(decoded)
        prob, pred = out.data.topk(1)
        prob = prob.squeeze(2)
        pred = pred.squeeze(2)
        # pred = torch.argmax(out, -1)

        return pred, hidden, attn

    # TODO
    def beam_translate(self):
        # beam translate
        raise NotImplementedError


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dim, type='general'):
        super(Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.type = type

        if type == 'general':
            # score : (tar_seq, dim) x (dim, dim) x (dim, src_seq) => tar_seq, src_seq
            # ctx   : (tar_seq, src_seq) x (src_seq, dim) => tar_seq, dim

            # self.weight = Variable(torch.FloatTensor(self.hidden_dim, self.hidden_dim), requires_grad=True)
            self.weight = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim, dtype=torch.float))
        elif type == 'dot':
            # score : (tar_seq, dim) x (dim, src_seq) => tar_seq, src_seq
            # ctx   : (tar_seq, src_seq) x (src_seq, dim) => tar_seq, dim
            pass
        elif type == 'concat':
            # score : (tar_seq, src_seq, dim * 2) x (dim * 2, attn_dim) x (attn_dim, 1) => tar_seq, src_seq
            # ctx   : (tar_seq, src_seq) x (src_seq, dim) => tar_seq, dim

            # self.weight = Variable(torch.FloatTensor(self.hidden_dim * 2, self.attn_dim), requires_grad=True)
            # self.weight_2 = Variable(torch.FloatTensor(self.attn_dim, 1), requires_grad=True)

            self.weight = nn.Parameter(torch.randn(self.hidden_dim * 2, self.attn_dim, dtype=torch.float))
            self.weight_2 = nn.Parameter(torch.randn(self.attn_dim, 1, dtype=torch.float))
        else:
            raise NotImplementedError

        self.proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(-1)
        self.sm = nn.Softmax(-1)


    def forward(self, enc, dec):
        # enc   : batch, src_seq, dim
        # dec   : batch, tar_seq, dim

        batch, src_seq, dim = enc.shape
        batch, tar_seq, dim = dec.shape

        # Compute Score
        if self.type == 'general':
            weight_exp = self.weight.unsqueeze(0).expand(batch, self.hidden_dim, self.hidden_dim)

            affine = torch.bmm(dec, weight_exp)
            # affine = torch.bmm(dec, self.weight)
            affine = torch.bmm(affine, enc.transpose(1, 2))
        elif self.type == 'dot':
            affine = torch.bmm(dec, enc.transpose(1, 2))
        elif self.type == 'concat':
            weight_exp = self.weight.unsqueeze(0).expand(batch, self.hidden_dim * 2, self.attn_dim)
            weight_2_exp = self.weight_2.unsqueeze(0).expand(batch, self.attn_dim, 1)

            enc_exp = enc.unsqueeze(1).expand(batch, tar_seq, src_seq, dim)
            dec_exp = dec.unsqueeze(2).expand(batch, tar_seq, src_seq, dim)
            concat = torch.cat((enc_exp, dec_exp), 3).view(batch, -1, dim * 2)
            affine = torch.bmm(concat, weight_exp)
            affine = torch.bmm(affine, weight_2_exp).view(batch, tar_seq, src_seq)
            # affine = torch.bmm(concat, self.weight)
            # affine = torch.bmm(affine, self.weight_2).view(batch, tar_seq, src_seq)

        # Softmax
        # score = self.sm(affine)
        score = self.log_softmax(affine)

        # Get Context Vectors
        ctx = torch.bmm(score, enc)

        # Output Proj
        attn_hidden = torch.cat((ctx, dec), 2)

        return self.proj(attn_hidden), score