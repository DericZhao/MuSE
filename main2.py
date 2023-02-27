import math

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.nn.init import xavier_uniform_

from models import config
from models.utils import *
from models.Transformer.EmpGraphTrans import EmpGraphTrans
from models.GraphLoader1 import prepare_data_seq


def make_model(vocab, dec_num):
    is_eval = config.test
    if config.model == "emp_graph":
        model = EmpGraphTrans(
            vocab,
            decoder_number=dec_num,
            is_eval=is_eval,
            model_file_path=config.model_path if is_eval else None,
            is_multitask=True
        )
        
    model.to(config.device)

    # Intialization
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("# PARAMETERS", count_parameters(model))

    return model


def train(model, train_set, dev_set, test_set):
    patient = 0
    best_ppl = 1000

    for epoch in range(1, config.epochs + 1):
        print('-'*88)
        model.train()
        train_loss = []
        train_acc = []
        train_bar = tqdm(train_set)
        for data_iter in train_bar:
            loss, ppl, bce, acc = model.train_one_batch(data_iter)
            train_loss.append(loss)
            train_acc.append(acc)
            train_bar.set_description(
                "Epoch {:.0f} train loss {:.4f} ppl {:.2f} acc {:.4f}".format(epoch, np.mean(train_loss), math.exp(np.mean(train_loss)), np.mean(train_acc))
            )

        # evaluate in each epoch
        model.eval()
        valid_loss, ppl_val, bce_val, acc_val, _ = evaluate(model, dev_set, epoch, ty="valid", max_dec_step=50)

        print(f'Epoch {epoch} '
              f'train loss {round(float(np.mean(train_loss)), 4)} train ppl {round(math.exp(np.mean(train_loss)), 4)}, '
              f'train acc {round(float(np.mean(train_acc)), 4)}, '
              f'valid loss {round(float(valid_loss), 4)} valid ppl {round(ppl_val, 4)}, '
              f'valid acc {round(float(acc_val), 4)}, '
              )

        if ppl_val <= best_ppl:
            best_ppl = ppl_val
            patient = 0
            model.save_model(best_ppl, epoch)
            weights_best = deepcopy(model.state_dict())
            # execute testing program after 5 epochs
            if epoch >= 5:
                model.load_state_dict({name: weights_best[name] for name in weights_best})
                test(model, test_set, epoch)

        else:
            patient += 1

        if patient > 8:
            break


def evaluate(model, data, epoch, ty="valid", max_dec_step=30):
    model.__id__logger = 0
    ref, hyp_g, results = [], [], []
    if ty == "test":
        print("Testing Generation:")
    l = []
    p = []
    bce = []
    acc = []
    top_preds = []
    comet_res = []
    pbar = tqdm(enumerate(data), total=len(data))

    # t = Translator(model, model.vocab)
    for j, batch in pbar:
        loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, train=False)
        l.append(loss)
        p.append(ppl)
        bce.append(bce_prog)
        acc.append(acc_prog)
        if ty == "test":
            sent_g = model.decoder_greedy(batch, max_dec_step=max_dec_step)
            # sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
            for i, greedy_sent in enumerate(sent_g):
                rf = " ".join(batch["target_txt"][i])
                hyp_g.append(greedy_sent)
                ref.append(rf)
                temp = print_custum(
                    emotion=batch["key_emotion_text"][i],
                    dial=' '.join([word for sentence in batch["input_txt"][i] for word in sentence]),
                    ref=rf,
                    hyp_g=greedy_sent,
                    pred_emotions=top_preds,
                    comet_res=comet_res,
                )
                results.append(temp)
        if ty =="test":
            pbar.set_description(
                "Epoch {:.0f} test loss: {:.4f} ppl:{:.1f} acc: {:.4f}".format(epoch, np.mean(l), math.exp(np.mean(l)), np.mean(acc))
            )
        else:
            pbar.set_description(
                "Epoch {:.0f} valid loss: {:.4f} ppl:{:.1f} acc: {:.4f}".format(epoch, np.mean(l), math.exp(np.mean(l)), np.mean(acc))
            )

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)

    # print("EVAL\tLoss\tPPL\tAccuracy\n")
    # print("{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ty, loss, math.exp(loss), acc))

    return loss, math.exp(loss), bce, acc, results


def test(model, test_set, epoch):
    model.eval()
    model.is_eval = True
    loss_test, ppl_test, bce_test, acc_test, results = evaluate(
        model, test_set, epoch, ty="test", max_dec_step=50
    )
    file_summary = config.save_path + "/results.txt"
    with open(file_summary, "w", encoding='utf-8') as f:
        f.write("EVAL\tLoss\tPPL\tAccuracy\n")
        f.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                loss_test, ppl_test, bce_test, acc_test
            )
        )
        for r in results:
            f.write(r)


if __name__ == "__main__":
    # for reproducibility
    set_seed()

    train_set, dev_set, test_set, vocab, dec_num = prepare_data_seq(batch_size=config.batch_size)

    model = make_model(vocab, dec_num)

    train(model, train_set, dev_set, test_set)

    # for batch in train_set:
    #     # print(batch["input_batch"])
    #     # print(batch["input_batch"].shape)
    #     enc_batch = batch["input_batch"]
    #     mask_input = batch["mask_input"]
    #     print(mask_input.size(-1))
    #     print(enc_batch[:, 6, :].shape)
    #
    #
    #     break

