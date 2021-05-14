import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import math
from Models.dataprocess import dataprocess, normalizeString

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
teacher_forcing_ratio = 0.5
MAX_LENGTH = 384
max_answer_length = 10
SOS_token = 0
EOS_token = 1

### 数据处理，得到词表
train_lang, dev_lang = dataprocess("train", "dev")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, length):
    indexes = indexesFromSentence(lang, sentence)
    while len(indexes)<length:
        indexes.append(0)
    if len(indexes) > length:
        indexes = indexes[:length]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0).view(-1, 1, 1)


def tensorsFromPair(answers, length, lang):
    targets = []
    for answer in answers:
        targets.append(tensorFromSentence(lang, answer, length))
    target_tensor = torch.cat(targets, 1)
    return target_tensor

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, ouput_size, batch_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = ouput_size+1
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.batch_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        """
            input: 16 * 1
            hidden: 16 * 1 * 768
            encoder_outputs: 16 * 384 * 768
        """
        embedded = self.embedding(input).view(batch_size, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 2)), dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output), dim=2)
        # print("attn_weights: ", attn_weights.size())
        # print("hidden: ", hidden.size())
        # print("output: ", output.size())
        # exit(0)
        """
        output: [16, 1, 768]
        hidden: [16, 1, 768]
        attn_weights: [16, 1, 384]
        """

        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, 1, self.hidden_size, device=device)


def train(batch_size, target_tensor, encoder_output, decoder,
        decoder_optimizer, criterion, max_length=MAX_LENGTH):

    seq_len = encoder_output.size(1)
    hidden = encoder_output.view(seq_len, batch_size, -1)
    decoder_hidden = hidden[0].view(batch_size, 1, -1)

    decoder_optimizer.zero_grad()
    target_length = target_tensor.size(0)

    loss = 0

    ### encoder_output:
    #   seq_length * hidden_size
    #   384 * 768

    decoder_input = torch.zeros([batch_size, 1], dtype=torch.long, device=device) # 16 * 0

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # print("decoder_input: ", decoder_input)
    # print('encoder_outputs: ', encoder_output)
    # print('decoder_hidden: ', decoder_hidden)
    # exit(0)
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):

            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)
            # print("decoder_output: ", decoder_output)
            # print("decoder_hidden: ", decoder_hidden)
            # print("target_tensor: ", target_tensor.size())

            for i in range(batch_size):
                loss += criterion(decoder_output[i], target_tensor[di][i])
            decoder_input = target_tensor[di]  # Teacher forcing
        # print("decoder_output: ", decoder_output[-1])
        # print("target_tensor: ", target_tensor)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            for i in range(batch_size):
                loss += criterion(decoder_output[i], target_tensor[di][i])
                if decoder_input[i].item() == EOS_token:
                    break
    loss /= batch_size
    # loss.backward()

    # decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder_output, decoder, answer_strs, train_lang, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    batch_size = encoder_output.size(0)

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    target_tensor = tensorsFromPair(answer_strs, max_answer_length, train_lang)
    criterion = nn.NLLLoss()

    # print("batch_size: ", batch_size)
    # print("target_tensor: ", target_tensor.size())
    # print("encoder_output: ", encoder_output.size())

    loss = train(batch_size, target_tensor, encoder_output,
                 decoder, decoder_optimizer, criterion)
    return loss
    # print_loss_total += loss
    # plot_loss_total += loss

    # if iter % print_every == 0:
    #     print_loss_avg = print_loss_total / print_every
    #     print_loss_total = 0
    #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
    #                                  iter, iter / n_iters * 100, print_loss_avg))
    #
    # if iter % plot_every == 0:
    #     plot_loss_avg = plot_loss_total / plot_every
    #     plot_losses.append(plot_loss_avg)
    #     plot_loss_total = 0

    # showPlot(plot_losses)

def evaluate(encoder_output, decoder, sentence, max_length=max_answer_length):
    with torch.no_grad():
        batch_size = encoder_output.size(0)

        decoder_input = torch.zeros([batch_size, 1], dtype=torch.long, device=device)  # 16 * 0

        seq_len = encoder_output.size(1)
        hidden = encoder_output.view(seq_len, batch_size, -1)
        decoder_hidden = hidden[0].view(batch_size, 1, -1)

        decoded_words = [[] for i in range(batch_size)]
        decoder_attentions = torch.zeros(max_length, batch_size, MAX_LENGTH)

        """
               decoder_output: [16, 1, 14849]
               decoder_hidden: [16, 1, 768]
               decoder_attention: [16, 1, 384]
        """

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output)

            decoder_attentions[di] = decoder_attention.view(1,batch_size,-1).data

            for i in range(batch_size):
                topv, topi = decoder_output[i].data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words[i].append('<EOS>')
                    break
                else:
                    decoded_words[i].append(dev_lang.index2word[topi.item()])
                decoder_input[i] = topi.squeeze().detach()

        return decoded_words, decoder_attentions.view(batch_size, max_length, MAX_LENGTH)

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def seq2seq(encoder_output, answer_strs, context_str, updates):
    hidden_size = 768
    batch_size = encoder_output.size(0)
    attn_decoder1 = AttnDecoderRNN(hidden_size, dev_lang.n_words, batch_size, dropout_p=0.1).to(device)
    for i in range(len(answer_strs)):
        answer_strs[i] = normalizeString(answer_strs[i])
    loss = trainIters(encoder_output, attn_decoder1, answer_strs, train_lang)
    #
    for i in range(len(context_str)):
        context_str[i] = normalizeString(" ".join(context_str[i]))
    if updates > 0 and self.updates % 1600 == 0:
        output_words, attentions = evaluate(
            encoder_output, attn_decoder1, context_str)
        print('answer =', answer_strs)
        print('output =', output_words)

    return loss