### TAKEN FROM https://github.com/kolloldas/torchnlp

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
import dgl

import networkx as nx
import matplotlib.pyplot as plt


import numpy as np
import math
from models.common import (
    EncoderLayer,
    DecoderLayer,
    LayerNorm,
    _gen_bias_mask,
    _gen_timing_signal,
    share_embedding,
    LabelSmoothing,
    NoamOpt,
    _get_attn_subsequent_mask,
    get_input_from_batch,
    get_output_from_batch,
    top_k_top_p_filtering,
)
from models import config
from sklearn.metrics import accuracy_score


class Encoder(nn.Module):
    """
    A Transformer Encoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        use_mask=False,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size) # []1, 1000, 300

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (
            hidden_size, # 300
            total_key_depth or hidden_size, # 40
            total_value_depth or hidden_size, # 40
            filter_size, # 50
            num_heads,# 2
            _gen_bias_mask(max_length) if use_mask else None, # None
            layer_dropout, # 0.0
            attention_dropout, # 0.0
            relu_dropout, # 0.0
        )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False) # 预训练词嵌入映射
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)


    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.enc,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                )
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, : inputs.shape[1], :].type_as(
                        inputs.data
                    )
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    """
    A Transformer Decoder module.
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        embedding_size,
        hidden_size,
        num_layers,
        num_heads,
        total_key_depth,
        total_value_depth,
        filter_size,
        max_length=1000,
        input_dropout=0.0,
        layer_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        universal=False,
    ):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(max_length)

        params = (
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            _gen_bias_mask(max_length),  # mandatory
            layer_dropout,
            attention_dropout,
            relu_dropout,
        )

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                *[DecoderLayer(*params) for l in range(num_layers)]
            )

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(
            mask_trg + self.mask[:, : mask_trg.size(-1), : mask_trg.size(-1)], 0
        )
        # Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(
                    x,
                    inputs,
                    self.dec,
                    self.timing_signal,
                    self.position_signal,
                    self.num_layers,
                    encoder_output,
                    decoding=True,
                )
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += (
                        self.position_signal[:, l, :]
                        .unsqueeze(1)
                        .repeat(1, inputs.shape[1], 1)
                        .type_as(inputs.data)
                    )
                    x, _, attn_dist, _ = self.dec(
                        (x, encoder_output, [], (mask_src, dec_mask))
                    )
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, : inputs.shape[1], :].type_as(inputs.data)

            # Run decoder
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

            # Final layer normalization
            y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        x,
        attn_dist=None,
        enc_batch_extend_vocab=None,
        extra_zeros=None,
        temp=1,
        beam_search=False,
        attn_dist_db=None,
    ):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        logit = self.proj(x)

        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat(
                [enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1
            )  ## extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat(
                    [enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0
                )  ## extend for all seq
            logit = torch.log(
                vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_)
            )
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


def Graph_Method(batch, embedding, encoder, encoder_outputs, graph, graph_batch=False):
    key_situation_batch = batch["key_situation_batch"]
    mask_key_situation = key_situation_batch.data.eq(config.PAD_idx).unsqueeze(1)
    key_situation_mask = embedding(batch["key_situation_mask"])
    key_situation_outputs = encoder(embedding(key_situation_batch) + key_situation_mask, mask_key_situation)

    G_batch = []
    local_state_G = []
    local_state_POS = []

    # building batch_graph
    for bat, emotion in enumerate(batch["context_emotion"]):
        src = []
        dst = []
        # key emotion = 0
        # key situation = 1
        key_emotion_index = [0]
        key_situation_index = [1]
        emotion_index = [index for index in range(2, len(emotion) + 2)]
        utterance_index = [index for index in range(len(emotion) + 2, 2 * len(emotion) + 2)]
        local_state_index = [index for index in
                             range(2 * len(emotion) + 2, 2 * len(emotion) + 2 + len(batch["Speaker_CLS_POS"][bat]))]

        # add key emotion to emotion edge
        for ke in range(len(emotion_index)):
            src.append(key_emotion_index[0])
            dst.append(emotion_index[ke])

        # add emotion to utterance edge
        for ei in range(len(utterance_index)):
            src.append(emotion_index[ei])
            dst.append(utterance_index[ei])

        # add situation, utterance, emotion to local emotion state edge
        count = -1
        for local_state in range(len(batch["Speaker_CLS_POS"][bat])):
            for _ in range(batch["Speaker_CLS_POS"][bat][local_state]):
                count += 1
                # add emotion to local emotion state edge
                src.append(emotion_index[count])
                dst.append(local_state_index[local_state])

                # add utterance to local emotion state edge
                src.append(utterance_index[count])
                dst.append(local_state_index[local_state])

            # add key situation to local emotion state edge
            src.append(key_situation_index[0])
            dst.append(local_state_index[local_state])

            # add local emotion state to next local emotion state
            if len(batch["Speaker_CLS_POS"][bat]) == 1:
                pass
            elif local_state == len(batch["Speaker_CLS_POS"][bat]) - 1:
                pass
            elif local_state == len(batch["Speaker_CLS_POS"][bat]) - 2:
                src.append(local_state_index[local_state])
                dst.append(local_state_index[local_state + 1])
            else:
                src.append(local_state_index[local_state])
                dst.append(local_state_index[local_state + 1])

                src.append(local_state_index[local_state])
                dst.append(local_state_index[local_state + 2])

        # u = np.concatenate([src, dst])
        # v = np.concatenate([dst, src])

        G = dgl.graph((src, dst))
        G = dgl.add_self_loop(G)
        G = G.to(config.device)
        # key_emotion_feature = self.embedding(batch["key_emotion"][bat].to(config.device)).unsqueeze(0)
        # key_emotion_feature = embedding(torch.LongTensor([config.KEY_EMOTION]).to(config.device))
        key_emotion_feature = encoder_outputs[bat, batch["CLS_POS"][bat]][0].unsqueeze(0)

        key_situation_feature = key_situation_outputs[bat, 0].unsqueeze(0)
        # key_situation_feature = encoder_outputs[bat, batch["CLS_POS"][bat]][0].unsqueeze(0)
        # key_situation_feature = embedding(torch.LongTensor([config.KEY_SITUATION]).to(config.device))

        emotion_feature = embedding(emotion.to(config.device))
        utterance_feature = encoder_outputs[bat, batch["CLS_POS"][bat]]
        feature = torch.concat((key_emotion_feature, key_situation_feature, emotion_feature, utterance_feature), dim=0)

        count = -1
        for local_state in range(len(batch["Speaker_CLS_POS"][bat])):
            local_mean = []
            for _ in range(batch["Speaker_CLS_POS"][bat][local_state]):
                count += 1
                local_mean.append(utterance_feature[count])
            local_mean_feature = torch.mean(torch.stack(local_mean), dim=0).unsqueeze(0)
            feature = torch.concat((feature, local_mean_feature), dim=0)

        G.ndata['feature'] = feature
        G_batch.append(G)
        if not graph_batch:
            gat_output = graph(G, G.ndata['feature'])
            # mean graph
            encoder_outputs[bat, 0] = torch.mean(gat_output[local_state_index, ], dim=0)

            # ffd graph node
            # ffd = nn.Linear(len(gat_output[local_state_index, ]) * 300, 300).to(config.device)
            # encoder_outputs[bat, 0] = ffd(gat_output[local_state_index, ].view(-1))
        # print(G)
        else:
            if len(local_state_G) == 0:
                local_state_G.extend(local_state_index)
            else:
                num_node = local_state_G[-1]
                for index in local_state_index:
                    local_state_G.extend([index + num_node + 1])
            local_state_POS.append(len(local_state_index))

    if graph_batch:
        graph_batch = dgl.batch(G_batch)
        graph_batch = graph_batch.to(config.device)
        gat_output = graph(graph_batch, graph_batch.ndata['feature'])
        for bat, pos in enumerate(local_state_POS):
            encoder_outputs[bat, 0] = torch.mean(gat_output[sum(local_state_POS[: bat]): sum(local_state_POS[: bat+1]), ], dim=0)


class EmpGraphTrans(nn.Module):
    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False, is_multitask=False,):
        super(EmpGraphTrans, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.multitask = is_multitask

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = Encoder(
            config.emb_dim,
            config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
            universal=config.universal,
        )

        # self.graph = GAT(
        #     in_size=config.emb_dim,
        #     hid_size=config.hidden_dim,
        #     out_size=config.emb_dim,
        #     heads=[8, 1])

        self.graph = GCN(
            in_size=config.emb_dim,
            hid_size=config.hidden_dim,
            out_size=config.emb_dim,
        )

        # multiple decoders
        self.decoder = Decoder(
            config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.hop,
            num_heads=config.heads,
            total_key_depth=config.depth,
            total_value_depth=config.depth,
            filter_size=config.filter,
        )
        # 多任务 加上纬度映射 32个emotion纬度映射
        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        # 根据隐藏层进行生成句子
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(
                size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1
            )
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

        # 加快实现模型优化速率
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim,
                1,
                8000,
                torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            )

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=config.device)
            self.load_state_dict(state["model"])
            if load_optim:
                self.optimizer.load_state_dict(state["optimizer"])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter):
        state = {
            "iter": iter,
            "optimizer": self.optimizer.state_dict(),
            "current_loss": running_avg_ppl,
            "model": self.state_dict(),
        }
        model_save_path = os.path.join(
            self.model_dir,
            "TRS_{}_{:.4f}".format(iter, running_avg_ppl),
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, train=True):
        # enc_batch = [batch_size, length] 16, 7, 56
        enc_batch = batch["input_batch"]
        # enc_lens = [batch]
        enc_lens = batch["input_lengths"]
        cls_pos = batch["CLS_POS"]
        # dec_batch = [batch_size, length]
        dec_batch = batch["target_batch"]

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Encode
        # enc_batch.data.shape = [batch_size, length] 16, 110
        # mask_src = [batch_size, 1, max_turns, length] 16, 1, 7, 56
        # 把PAD符号 挑选出来
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        # batch["mask_input"] = [batch_size, max_turns, length] 16, 7, 56
        # emb_mask = [batch_size, max_turns, max_length, emb_dim]
        emb_mask = self.embedding(batch["mask_input"])
        # encoder_outputs = [batch_size, ]
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        Graph_Method(batch, self.embedding, self.encoder, encoder_outputs, self.graph)

        # Decode
        # sos_token = [batch_size, 1]
        sos_token = (
            torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        ).to(config.device)
        # dec_batch_shift = [batch_size, target_len] 中间扣除了最后的EOS标签
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        # mask_trg = [batch_size, 1, target_len]
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        # attn_dist = [batch_size, target_len, emb_dim]
        # pre_logit = [batch_size, target_len, emb_dim]
        pre_logit, attn_dist = self.decoder(
            self.embedding(dec_batch_shift), encoder_outputs, (mask_src, mask_trg)
        )

        # compute output dist
        # logit = [batch_size, target_len, vocab_size]
        logit = self.generator(
            pre_logit,
            attn_dist,
            attn_dist_db=None,
        )
        # logit = F.log_softmax(logit,dim=-1) #fix the name later
        ## loss: NNL if ptr else Cross entropy
        loss = self.criterion(
            logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)
        )

        # multi-task
        if self.multitask:
            # q_h = torch.mean(encoder_outputs,dim=1)
            q_h = encoder_outputs[:, 0]
            logit_prob = self.decoder_key(q_h)
            loss = self.criterion(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ) + 2 * nn.CrossEntropyLoss()(
                logit_prob, torch.LongTensor(batch["program_label"]).to(config.device)
            )
            loss_bce_program = nn.CrossEntropyLoss()(
                logit_prob, torch.LongTensor(batch["program_label"]).to(config.device)
            ).item()
            pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
            program_acc = accuracy_score(batch["program_label"], pred_program)

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(
                logit.contiguous().view(-1, logit.size(-1)),
                dec_batch.contiguous().view(-1),
            ).item()

        if train:
            loss.backward()
            self.optimizer.step()
        if self.multitask:
            if config.label_smoothing:
                return (
                    loss_ppl,
                    math.exp(min(loss_ppl, 100)),
                    loss_bce_program,
                    program_acc,
                )
            else:
                return (
                    loss.item(),
                    math.exp(min(loss.item(), 100)),
                    loss_bce_program,
                    program_acc,
                )
        else:
            if config.label_smoothing:
                return loss_ppl, math.exp(min(loss_ppl, 100)), 0, 0
            else:
                return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        # enc_batch = [batch_size, max_turns, length] 16, 7, 56
        enc_batch = batch["input_batch"]
        # enc_lens = [batch]
        enc_lens = batch["input_lengths"]

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        Graph_Method(batch, self.embedding, self.encoder, encoder_outputs, self.graph)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    (mask_src, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), encoder_outputs, (mask_src, mask_trg)
                )

            prob = self.generator(
                out, attn_dist, attn_dist_db=None
            )
            # logit = F.log_softmax(logit,dim=-1) #fix the name later
            # filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=0, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            # next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        (
            enc_batch,
            _,
            _,
            enc_batch_extend_vocab,
            extra_zeros,
            _,
            _,
            _,
        ) = get_input_from_batch(batch)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        Graph_Method(batch, self.embedding, self.encoder, encoder_outputs, self.graph)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(
                    self.embedding_proj_in(self.embedding(ys)),
                    self.embedding_proj_in(encoder_outputs),
                    (mask_src, mask_trg),
                )
            else:
                out, attn_dist = self.decoder(
                    self.embedding(ys), encoder_outputs, (mask_src, mask_trg)
                )

            logit = self.generator(
                out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None
            )
            filtered_logit = top_k_top_p_filtering(
                logit[:, -1], top_k=3, top_p=0, filter_value=-float("Inf")
            )
            # Sample from the filtered distribution
            next_word = torch.multinomial(
                F.softmax(filtered_logit, dim=-1), 1
            ).squeeze()
            decoded_words.append(
                [
                    "<EOS>"
                    if ni.item() == config.EOS_idx
                    else self.vocab.index2word[ni.item()]
                    for ni in next_word.view(-1)
                ]
            )
            next_word = next_word.data[0]

            ys = torch.cat(
                [ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)],
                dim=1,
            ).to(config.device)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ""
            for e in row:
                if e == "<EOS>":
                    break
                else:
                    st += e + " "
            sent.append(st)
        return sent


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()

        # total_Gat_layers = 8
        # for layers in range(total_Gat_layers-1):
        #     if layers == 0:
        #         self.gat_layers.append(
        #             dgl.nn.GATConv(
        #                 in_size,
        #                 hid_size,
        #                 heads[0],
        #                 feat_drop=0.6,
        #                 attn_drop=0.6,
        #                 activation=F.elu,
        #             )
        #         )
        #     else:
        #         self.gat_layers.append(
        #             dgl.nn.GATConv(
        #                 hid_size * heads[layers],
        #                 hid_size,
        #                 heads[layers],
        #                 feat_drop=0.6,
        #                 attn_drop=0.6,
        #                 activation=F.elu,
        #             )
        #         )
        # self.gat_layers.append(
        #     dgl.nn.GATConv(
        #         hid_size * heads[0],
        #         out_size,
        #         heads[1],
        #         feat_drop=0.6,
        #         attn_drop=0.6,
        #         activation=None,
        #     )
        # )

        # two-layer GAT
        self.gat_layers.append(
            dgl.nn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )

        self.gat_layers.append(
            dgl.nn.GATConv(
                hid_size * heads[0],
                out_size,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            # print(i)
            # print(h.shape)
            h = layer(g, h)
            # print(h.shape)
            if i == len(self.gat_layers)-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dgl.nn.GraphConv(in_size, hid_size, activation=F.relu)
        )

        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size))
        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size))
        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size))
        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size))
        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size))
        # self.layers.append(dgl.nn.GraphConv(hid_size, hid_size))

        self.layers.append(dgl.nn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h