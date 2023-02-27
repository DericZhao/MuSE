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

# classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=1, device=0)
def emo_caught(input, vocab):
    classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=1, device=0)
    vocab.index_word(classifier(input)[0][0]["label"])
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
        "context_emotion": [],  # 通过Roberta制造的上下文情绪标签
        "target_emotion": []  # 回答中的情绪
    }

    for i, k in enumerate(data_dict.keys()):
        items = files[i]
        if k == "context":
            for context in tqdm(items):
                # 两个说话人在一个context中的分割话语(utterance)
                speaker_turn = []
                speaker_turn_emotion = []
                for speaker in context:
                    # 预处理
                    speaker = process_sent(speaker)
                    # 构建词表
                    vocab.index_words(speaker)
                    # 一个speaker的所有分割话语
                    utterance = []
                    utterance_emotion = []
                    if speaker.count('.') + speaker.count('!') + speaker.count('?') > 0:
                        dot_index = [i for i, word in enumerate(speaker) if word == '.']
                        exclamation_index = [i for i, word in enumerate(speaker) if word == '!']
                        question_index = [i for i, word in enumerate(speaker) if word == '?']

                        # 统计所有的分割点的位置
                        split_index = dot_index + exclamation_index + question_index
                        split_index.sort()
                        # print(split_index)
                        for i, index in enumerate(split_index):
                            if i == 0:
                                utterance.append(speaker[:index + 1])
                                utterance_emotion.append(emo_caught(' '.join(speaker[:index + 1]), vocab))
                            else:
                                utterance.append(speaker[split_index[i - 1] + 1: index + 1])
                                utterance_emotion.append(emo_caught(' '.join(speaker[split_index[i - 1] + 1: index + 1]), vocab))
                        # 如果最后一个分割点不是句子的最后一个单词 那么再补上 如果是最后一个单词 不补（这样的情况就是一个空列表）
                        if len(speaker[split_index[-1] + 1:]) != 0:
                            utterance.append(speaker[split_index[-1] + 1:])
                            utterance_emotion.append(emo_caught(' '.join(speaker[split_index[-1] + 1:]), vocab))
                    else:
                        utterance.append(speaker)
                        utterance_emotion.append(emo_caught(' '.join(speaker), vocab))
                    speaker_turn.append(utterance)
                    speaker_turn_emotion.append(utterance_emotion)
                    if len(utterance) != len(utterance_emotion):
                        print(utterance)
                        print(utterance_emotion)
                    assert len(utterance) == len(utterance_emotion)
                    assert len(speaker) == len([word for utt in utterance for word in utt])

                data_dict["context"].append(speaker_turn)
                data_dict["context_emotion"].append(speaker_turn_emotion)
                assert len(speaker_turn) == len(speaker_turn_emotion)

        elif k == "emotion":
            for item in items:
                data_dict[k].append([item])

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
                data_dict["target_emotion"].append([emo_caught(' '.join(item), vocab)])

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
    cache_file = f"{data_dir}/dataset_emotionflow.more_emotion"
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

                    config.KEY_EMOTION: "KEY_EMOTION",
                    config.KEY_SITUATION: "KEY_SITUATION",
                }
            )
        )
        with open(cache_file, "wb") as f:
            pickle.dump([data_tra, data_val, data_tst, vocab], f)
            print("Saved PICKLE")

    for i in range(3):
        print("[context]:", [" ".join(word) for u in data_tra["context"][i] for word in u])
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
        item["key_situation_text"] = self.data["situation"][index]
        item["key_emotion_text"] = self.data["emotion"][index]

        item["context"], item["context_mask"], item['CLS_POS'], item['Speaker_CLS_POS'] = self.preprocess(item["context_text"], mask=True)
        item["context_emotion"] = self.preprocess(item["context_emotion_text"], context_emotion=True)
        item["target"] = self.preprocess(item["target_text"])
        item["target_emotion"] = self.preprocess(item["target_emotion_text"], target_emotion=True)
        item["key_emotion"] = self.preprocess(item["key_emotion_text"], target_emotion=True)
        item["key_situation"], item["key_situation_mask"] = self.preprocess(item["key_situation_text"], situation=True)

        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["key_emotion_text"][0], self.emo_map)

        return item

    def preprocess(self, sentences, mask=False, situation=False, context_emotion=False, target_emotion=False):
        """
        Sentences is the input single context contains some utterance
        Converts words to ids.
        mask symbol for context
        emotion symbol for context
        """
        if mask:
            x_dial = [config.CLS_idx]
            x_mask = [config.CLS_idx]

            CLS_POS = [0]
            Speaker_CLS_POS = []
            for i, sentence in enumerate(sentences):
                Speaker_CLS_POS.append(len(sentence))
                for j, utterance in enumerate(sentence):
                    x_dial += [
                        self.vocab.word2index[word]
                        if word in self.vocab.word2index
                        else config.UNK_idx
                        for word in utterance
                    ]

                    spk = (
                        self.vocab.word2index["USR"]
                        if i % 2 == 0
                        else self.vocab.word2index["SYS"]
                    )

                    x_mask += [spk for _ in range(len(utterance))]
                    if i == len(sentences) - 1 and j == len(sentence) - 1:
                        pass
                    else:
                        x_dial += [config.CLS_idx]
                        x_mask += [config.CLS_idx]
                        CLS_POS.append(CLS_POS[-1] + len(utterance) + 1)

            assert len(x_dial) == len(x_mask)
            assert len(CLS_POS) == sum(Speaker_CLS_POS)

            return torch.LongTensor(x_dial), torch.LongTensor(x_mask), torch.LongTensor(CLS_POS), torch.LongTensor(Speaker_CLS_POS)

        elif context_emotion:
            context_emotion_sequence = []
            for sentence in sentences:
                for word in sentence:
                    if word in self.vocab.word2index:
                        context_emotion_sequence.append(self.vocab.word2index[word])
                    else:
                        context_emotion_sequence.append(context_emotion_sequence.append(config.UNK_idx))

            return torch.LongTensor(context_emotion_sequence)

        elif target_emotion:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in sentences
            ]

            return torch.LongTensor(sequence)

        elif situation:
            situation_dial = [config.CLS_idx]
            situation_mask = [config.CLS_idx]

            spk = (self.vocab.word2index["USR"])
            situation_mask += [spk for _ in range(len(sentences))]

            situation_dial += [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in sentences
            ]

            return torch.LongTensor(situation_dial), torch.LongTensor(situation_mask)

        else:
            sequence = [
                self.vocab.word2index[word]
                if word in self.vocab.word2index
                else config.UNK_idx
                for word in sentences
            ] + [config.EOS_idx]

            return torch.LongTensor(sequence)

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]


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

    # sort according to the turns, the first is the largest turns
    data.sort(key=lambda x: len(x["context"]), reverse=True)

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # context and target
    input_batch, input_lengths = merge(item_info["context"])
    mask_input, mask_input_lengths = merge(item_info["context_mask"])

    situation_batch, situation_lengths = merge(item_info["key_situation"])
    mask_situation, mask_situation_lengths = merge(item_info["key_situation_mask"])

    target_batch, target_lengths = merge(item_info["target"])


    # emotion
    # context_emotion, _ = merge(item_info["context_emotion"])

    input_batch = input_batch.to(config.device)
    mask_input = mask_input.to(config.device)

    situation_batch = situation_batch.to(config.device)
    mask_situation = mask_situation.to(config.device)


    target_batch = target_batch.to(config.device)

    # Return the data required by model
    batch_data = {}
    # context and target
    batch_data["input_batch"] = input_batch
    batch_data["input_lengths"] = input_lengths
    batch_data["mask_input"] = mask_input

    batch_data["key_situation_batch"] = situation_batch
    batch_data["situation_lengths"] = situation_lengths
    batch_data["key_situation_mask"] = mask_situation

    batch_data["target_batch"] = target_batch
    batch_data["target_lengths"] = torch.LongTensor(target_lengths)

    # emotion
    batch_data["CLS_POS"] = item_info["CLS_POS"]
    batch_data["Speaker_CLS_POS"] = item_info["Speaker_CLS_POS"]
    batch_data["context_emotion"] = item_info["context_emotion"]
    batch_data["target_emotion"] = torch.LongTensor(item_info["target_emotion"])
    batch_data["key_emotion"] = torch.LongTensor(item_info["key_emotion"])
    batch_data["target_program"] = item_info["emotion"]
    batch_data["program_label"] = item_info["emotion_label"]

    # text
    batch_data["input_txt"] = [word for utterance in item_info["context_text"] for word in utterance]
    batch_data["target_txt"] = item_info["target_text"]
    batch_data["key_emotion_text"] = item_info["key_emotion_text"]
    batch_data["key_situation_text"] = item_info["key_situation_text"]

    return batch_data


def split_local_emotion_state(input_file):
    new_context = []
    for context in input_file:
        speaker_turn = []
        for speaker in context:
            utterance = []
            if speaker.count('.') + speaker.count('!') + speaker.count('?') > 0:
                dot_index = [i for i, word in enumerate(speaker) if word == '.']
                exclamation_index = [i for i, word in enumerate(speaker) if word == '!']
                question_index = [i for i, word in enumerate(speaker) if word == '?']

                split_index = dot_index + exclamation_index + question_index
                split_index.sort()
                # 统计所有的分割点的位置
                # print(split_index)
                for i, index in enumerate(split_index):
                    if i == 0:
                        utterance.append(speaker[:index + 1])
                    else:
                        utterance.append(speaker[split_index[i - 1] + 1: index + 1])
                # 如果最后一个分割点不是句子的最后一个单词 那么再补上 如果是最后一个单词 不补（这样的情况就是一个空列表）
                if len(speaker[split_index[-1] + 1:]) != 0:
                    utterance.append(speaker[split_index[-1] + 1:])
            else:
                utterance.append(speaker)
            speaker_turn.append(utterance)
        new_context.append(speaker_turn)
    return new_context


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





