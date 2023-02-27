import os
import nltk
import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from models import config
import torch.utils.data as Data
from models.utils import save_config
from models.config import DATA_FILES
from models.config import WORD_PAIRS as word_pairs
from transformers import pipeline
import dgl

# classifier = pipeline("text-classification", model="j-hartmann/emotion-english-roberta-large", top_k=1, device=0)
def emo_caught(input):
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-roberta-large", top_k=1, device=0)
    return classifier(input)[0][0]["label"]


def process_sent(sentence):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    sentence = nltk.word_tokenize(sentence)
    return sentence


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_files(vocab):
    files = DATA_FILES(config.data_dir)
    train_files = [np.load(f, allow_pickle=True) for f in files["train"]]
    dev_files = [np.load(f, allow_pickle=True) for f in files["dev"]]
    test_files = [np.load(f, allow_pickle=True) for f in files["test"]]

    data_train = encode(vocab, train_files)
    data_dev = encode(vocab, dev_files)
    data_test = encode(vocab, test_files)
    return data_train, data_dev, data_test, vocab


def encode(vocab, files):
    data_dict = {
        "context": [],# 上下文
        "target": [],  # 目标回答
        "emotion": [],  # 共情对话中的主要主题
        "situation": [],  # prompt
        "context_emotion": [],  #通过Roberta制造的上下文情绪标签
        "target_emotion": [],# 对话中的情绪
    }

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            # encode_ctx(vocab, items, data_dict)
            for item in tqdm(items):
                ctx_list = []
                ctx_emo_list = []
                for context in item:
                    context = process_sent(context)
                    vocab.index_words(context)
                    ctx_list.append(context)
                    ctx_emo_list.append(emo_caught(' '.join(context)))
                data_dict["context"].append(ctx_list)
                data_dict["context_emotion"].append(ctx_emo_list)

        elif k == "emotion":
            data_dict[k] = items

        elif k == "situation":
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)

        else:
            for item in tqdm(items):
                item = process_sent(item)
                data_dict[k].append(item)
                vocab.index_words(item)
                data_dict["target_emotion"].append([emo_caught(' '.join(item))])

        if i == 3:
            break

    assert (
        len(data_dict["context"])
        == len(data_dict["context_emotion"])
        == len(data_dict["target"])
        == len(data_dict["target_emotion"])
        == len(data_dict["emotion"])
        == len(data_dict["situation"])
    )

    return data_dict


def load_dataset():
    data_dir = config.data_dir
    cache_file = f"{data_dir}/dataset_emotionflow.p1"
    if os.path.exists(cache_file):
        print("LOADING empathetic_dialogue")
        with open(cache_file, "rb") as f:
            [data_tra, data_val, data_tst, vocab] = pickle.load(f)
    else:
        print("Building dataset...")
        data_tra, data_val, data_tst, vocab = read_files(
            vocab=Lang(
                {
                    config.UNK_idx: "UNK",
                    config.PAD_idx: "PAD",
                    config.EOS_idx: "EOS",
                    config.SOS_idx: "SOS",
                    config.USR_idx: "USR",
                    config.SYS_idx: "SYS",
                    config.CLS_idx: "CLS",

                    config.joy_idx: "joy",
                    config.sadness_idx: "sadness",
                    config.surprise_idx: "surprise",
                    config.neutral_idx: "neutral",
                    config.anger_idx: "anger",
                    config.fear_idx: "fear",
                    config.disgust_idx: "disgust",

                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(3):
        print("[context]:", [" ".join(u) for u in data_tra["context"][i]])
        print("[context_emotion]:", [u for u in data_tra["context_emotion"][i]])
        print("[target]:", " ".join(data_tra["target"][i]))
        print("[target_emotion]:", data_tra["target_emotion"][i])
        print("[situation]:", " ".join(data_tra["situation"][i]))
        print("[emotion]:", data_tra["emotion"][i])
        print(" ")
    return data_tra, data_val, data_tst, vocab


class Dataset(Data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        # data = [context, context_emotion, target, target_emotion, situation, emotion]
        self.vocab = vocab
        self.data = data
        self.emo_map = config.EMO_MAP

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = {}

        item["context_text"] = self.data["context"][index]
        item["context_emotion_text"] = self.data["context_emotion"][index]
        item["target_text"] = self.data["target"][index]
        item["target_emotion_text"] = self.data["target_emotion"][index]
        item["situation_text"] = self.data["situation"][index]
        item["emotion_text"] = self.data["emotion"][index]

        item["context"], item["context_mask"], item['CLS_POS'] = self.preprocess(item["context_text"], mask=True)
        item["context_emotion"] = self.preprocess(item["context_emotion_text"], emotion=True)
        item["target"] = self.preprocess(item["target_text"])
        item["target_emotion"] = self.preprocess(item["target_emotion_text"], emotion=True)
        item["situation"] = self.preprocess(item["situation_text"])
        item["emotion"] = self.preprocess(item["emotion_text"], emotion=True)

        return item

    def preprocess(self, sentences, mask=False, emotion=False):
        """Converts words to ids."""
        if mask:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]
            CLS_POS = [0]
            for i, sentence in enumerate(sentences):
                x_dial += [
                    self.vocab.word2index[word]
                    if word in self.vocab.word2index
                    else config.UNK_idx
                    for word in sentence
                ]
                spk = (
                    self.vocab.word2index["USR"]
                    if i % 2 == 0
                    else self.vocab.word2index["SYS"]
                )
                x_mask += [spk for _ in range(len(sentence))]
                if i != len(sentences)-1:
                    x_dial += [config.CLS_idx]
                    x_mask += [config.CLS_idx]
                    CLS_POS.append(CLS_POS[-1] + len(sentence) + 1)
            assert len(x_dial) == len(x_mask)
            assert len(sentences) == len(CLS_POS)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask), torch.LongTensor(CLS_POS)

        elif emotion:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in sentences
            ]

            return torch.LongTensor(sequence)

        else:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in sentences
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)


def collate_fn(data):
    '''

    :param data: the origin data collected from the dataset
    :return: padding sequence required from the model
    '''
    # data = {context, context_emotion, target, target_emotion, situation, emotion}
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), max(lengths)
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    def h_merge(sequences):
        lengths = []
        max_lengths = []
        for sequence in sequences:
            length = [len(context) for context in sequence]
            lengths.append(length)
            max_lengths.extend(length)

        # padded_matrix = [batch_size, max_turns * max_lengths] padding index 1
        padded_matrix = torch.ones(len(sequences), config.max_turns, max(max_lengths)).long()

        for i, sequence in enumerate(sequences):
            for j, context in enumerate(sequence):
                end = lengths[i]
                padded_matrix[i, j, :end[j]] = context[:end[j]]

        return padded_matrix, lengths

    def graph_merge(sequences):# padding to max_turn
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(
            len(sequences), config.max_turns
        ).long()  ## padding index 1
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort according to the turns, the first is the largest turns
    data.sort(key=lambda x: len(x["context"]), reverse=True)

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])
    # emotion_input, _ = graph_merge(item_info["context_emotion"])
    # CLS_input, _ = graph_merge(item_info["CLS_POS"])

    target_batch, target_lengths = merge(item_info["target"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)
    # emotion_input = emotion_input.to(config.device)
    # CLS_input = CLS_input.to(config.device)
    target_batch = target_batch.to(config.device)

    # Return the data required by model
    batch_data = {}
    batch_data["input_batch"] = input_batch
    batch_data["input_lengths"] = input_lengths
    batch_data["mask_input"] = mask_input
    batch_data["context_emotion"] = item_info["context_emotion"]
    batch_data["CLS_POS"] = item_info["CLS_POS"]

    batch_data["target_batch"] = target_batch
    batch_data["target_lengths"] = torch.LongTensor(target_lengths)
    batch_data["target_emotion"] = torch.LongTensor(item_info["target_emotion"])

    # text
    batch_data["input_txt"] = item_info["context_text"]
    batch_data["target_txt"] = item_info["target_text"]
    batch_data["program_txt"] = item_info["emotion_text"]
    batch_data["situation_txt"] = item_info["situation_text"]

    return batch_data


def prepare_data_seq(batch_size=32):

    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    save_config()
    return (
        data_loader_tra,
        data_loader_val,
        data_loader_tst,
        vocab,
        len(dataset_train.emo_map)
    )

