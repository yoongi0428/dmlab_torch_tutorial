import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            input_size=self.emb_dim,
            hidden_size=self.rnn_hidden,
            num_layers=self.num_layers,
            bidirectional=self.bi,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.rnn_hidden,
            num_layers=self.num_layers,
            batch_first=True
        )
        enc_dim = rnn_hidden * 2 if bi else rnn_hidden
        if attn:
            self.attention = Attention(enc_dim, self.rnn_hidden, self.attn_dim, self.attn_type)

        self.hidden_bridge = nn.Linear(enc_dim, self.rnn_hidden)

        self.out_proj = nn.Linear(self.rnn_hidden, tar_vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax()
        self.log_sm = nn.LogSoftmax(-1)

    # TODO why this stupid model keeps generating <SOS>? NEED ANY HELP, YOU XXX?
    def forward(self, src, tar, teacher_forcing_ratio=0.5):
        src_emb = self.src_emb(src)

        encoded, hidden = self.encoder(src_emb)

        hidden = self._bridge(hidden)

        tar_emb = self.tar_emb(tar)

        decoded, _ = self.decoder(tar_emb, hidden)

        score = None
        if self.attn:
            enc_mask = src.eq(3)    # 3 = PAD
            decoded, score = self.attention(encoded, decoded, enc_mask)

        out = self.out_proj(decoded)

        return self.log_sm(out), score


    def _bridge(self, hidden):
        if hidden[0].size(0) == 2:
            hidden = tuple([torch.cat(x.split(1, 0), dim=-1) for x in hidden])
        h, c = tuple([self.hidden_bridge(x) for x in hidden])
        return (F.relu(h), F.relu(c))

    def translate(self, src, maxlen=50, SOS=0, EOS=1):
        # Greedy translate
        batch, src_seq = src.shape
        src_emb = self.src_emb(src)

        encoded, hidden = self.encoder(src_emb)

        hidden = self._bridge(hidden)

        translations = []
        attentions = []


        inp = torch.ones(batch, 1, dtype=torch.long) * SOS
        inp = inp.cuda()    # TODO
        for i in range(maxlen):
            # tar_inp : batch, cur_len, embedding
            tar_inp = self.tar_emb(inp)

            # decoded : batch, cur_len, embedding
            decoded, hidden = self.decoder(tar_inp, hidden)

            attn = None
            if self.attn:
                enc_mask = src.eq(3)
                decoded, attn = self.attention(encoded, decoded, enc_mask)
                attentions.append(attn)

            # decoded : batch, cur_len, output_dim
            out = self.out_proj(decoded)

            # pred : batch, cur_len, 1
            pred = torch.argmax(out, -1)
            inp = pred

            translations.append(pred)

            # pred = pred[:, -1].unsqueeze(1)
            # tar = torch.cat((tar, pred), 1)

        translations = torch.cat(translations, 1).tolist()
        if self.attn:
            ret = translations, torch.cat(attentions, 1).cpu().detach().numpy()
        else:
            ret = translations, None

        return ret




        # for b in range(batch):
        #     trans = []
        #     attn = []
        #     tar = torch.ones(1, 1, dtype=torch.long) * SOS
        #     cur_enc = encoded[b, :, :].unsqueeze(0)
        #     cur_hidden = tuple([h[:, 0, :].unsqueeze(1).contiguous() for h in hidden])
        #     for i in range(maxlen):
        #         pred, cur_hidden, one_attn = self.translate_one(cur_enc, tar.cuda(), cur_hidden)
        #
        #         if pred == EOS:
        #             break
        #
        #         trans.append(pred.item())
        #         attn.append(one_attn)
        #         tar = pred
        #
        #     translations.append(trans)
        #     attentions.append(attn)
        #
        # return translations, attentions

    def translate_batch(self, src, maxlen=50, SOS=0, EOS=1):
        # Greedy translate
        batch, src_seq = src.shape
        src_emb = self.src_emb(src)

        encoded, hidden = self.encoder(src_emb)

        translations = []
        attentions = []

        tar = torch.ones(1, 1, dtype=torch.long) * SOS
        tar = tar.cuda()
        attentions = torch.zeros(batch, maxlen, maxlen)
        for b in range(batch):
            cur_encoded = encoded[b, :, :].unsqueeze(0)
            cur_hidden = (hidden[0][:, b, :].unsqueeze(1), hidden[1][:, b, :].unsqueeze(1))
            for i in range(maxlen):
                # tar_inp : batch, cur_len, embedding
                tar_inp = self.tar_emb(tar)

                # decoded : batch, cur_len, embedding
                decoded, cur_hidden = self.decoder(tar_inp, cur_hidden)

                attn = None
                if self.attn:
                    decoded, attn = self.attention(cur_encoded, decoded)
                    attentions[b, :, :] += attn

                # decoded : batch, cur_len, output_dim
                out = self.out_proj(decoded)

                # pred : batch, cur_len, 1
                pred = torch.argmax(out, -1)
                pred = pred[:, -1].unsqueeze(1)

                tar = torch.cat((tar, pred), 1)

        return tar[:, 1:], attentions

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
    def __init__(self, enc_dim, dec_dim, attn_dim, type='general'):
        super(Attention, self).__init__()

        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = attn_dim
        self.type = type

        if type == 'general':
            # score : (tar_seq, dim) x (dim, dim) x (dim, src_seq) => tar_seq, src_seq
            # ctx   : (tar_seq, src_seq) x (src_seq, dim) => tar_seq, dim

            # self.weight = Variable(torch.FloatTensor(self.hidden_dim, self.hidden_dim), requires_grad=True)
            self.weight = nn.Parameter(torch.randn(self.dec_dim, self.enc_dim, dtype=torch.float), requires_grad=True)
        elif type == 'dot':
            # score : (tar_seq, dim) x (dim, src_seq) => tar_seq, src_seq
            # ctx   : (tar_seq, src_seq) x (src_seq, dim) => tar_seq, dim
            pass
        elif type == 'concat':
            # score : (tar_seq, src_seq, dim * 2) x (dim * 2, attn_dim) x (attn_dim, 1) => tar_seq, src_seq
            # ctx   : (tar_seq, src_seq) x (src_seq, dim) => tar_seq, dim

            # self.weight = Variable(torch.FloatTensor(self.hidden_dim * 2, self.attn_dim), requires_grad=True)
            # self.weight_2 = Variable(torch.FloatTensor(self.attn_dim, 1), requires_grad=True)

            self.weight = nn.Parameter(torch.randn(self.enc_dim + self.dec_dim, self.attn_dim, dtype=torch.float))
            self.weight_2 = nn.Parameter(torch.randn(self.attn_dim, 1, dtype=torch.float))
        else:
            raise NotImplementedError

        self.proj = nn.Linear(self.enc_dim + self.dec_dim, self.dec_dim)
        self.sm = nn.Softmax(-1)


    def forward(self, enc, dec, enc_mask=None):
        # enc   : batch, src_seq, dim
        # dec   : batch, tar_seq, dim

        batch, src_seq, enc_dim = enc.shape
        batch, tar_seq, dec_dim = dec.shape

        # Compute Score
        if self.type == 'general':
            weight_exp = self.weight.unsqueeze(0).expand(batch, self.dec_dim, self.enc_dim)

            affine = torch.bmm(dec, weight_exp)
            # affine = torch.bmm(dec, self.weight)
            affine = torch.bmm(affine, enc.transpose(1, 2))
        elif self.type == 'dot':
            affine = torch.bmm(dec, enc.transpose(1, 2))
        elif self.type == 'concat':
            weight_exp = self.weight.unsqueeze(0).expand(batch, self.enc_dim + self.dec_dim, self.attn_dim)
            weight_2_exp = self.weight_2.unsqueeze(0).expand(batch, self.attn_dim, 1)

            enc_exp = enc.unsqueeze(1).expand(batch, tar_seq, src_seq, enc_dim)
            dec_exp = dec.unsqueeze(2).expand(batch, tar_seq, src_seq, dec_dim)
            concat = torch.cat((enc_exp, dec_exp), 3).view(batch, -1, enc_dim + dec_dim)
            affine = torch.bmm(concat, weight_exp)
            affine = torch.bmm(affine, weight_2_exp).view(batch, tar_seq, src_seq)
            # affine = torch.bmm(concat, self.weight)
            # affine = torch.bmm(affine, self.weight_2).view(batch, tar_seq, src_seq)

        # batch, tar_seq, src_seq
        # mask : batch, srq_seq
        if enc_mask is not None:
            enc_mask = enc_mask.unsqueeze(1).repeat(1, tar_seq, 1)
            affine = affine.masked_fill(enc_mask, float('-inf'))

        # Softmax
        score = self.sm(affine)
        # score = self.log_softmax(affine)

        # Get Context Vectors
        ctx = torch.bmm(score, enc)

        # Output Proj
        attn_hidden = torch.cat((ctx, dec), 2)

        return F.relu(self.proj(attn_hidden)), score
        # return self.proj(attn_hidden), score