import torch
from torch.nn import GRU, Linear, ReLU
import torch.nn.functional as F
from torch_cddd.data import TOKENS
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from torch_cddd.modules import StackedDenseNet

PAD_VALUE = TOKENS.index("PAD")
NUM_TOKENS = len(TOKENS)

def gaussian_noise(input, training, stddev):
    if training:
        output = input + torch.randn(input.size()).cuda() * stddev
    else:
        output = input
    return output


class CDDDSeq2Seq(torch.nn.Module):
    def __init__(self, rnn_hidden_size, rnn_num_layers, emb_size, token_emb_size,
                 predictor_hidden_size, num_properties, input_noise_std, emb_noise_std,
                 input_dropout, decode_function=F.log_softmax):
        super().__init__()

        token_embedding = torch.nn.Embedding(len(TOKENS), token_emb_size)
        self.emb_noise_std = emb_noise_std
        self.encoder = EncoderRNN(
            vocab_size=len(TOKENS),
            max_len=None,
            hidden_size=rnn_hidden_size,
            n_layers=rnn_num_layers,
            variable_lengths=True,
            embedding=token_embedding,
            input_noise_std=input_noise_std,
            input_dropout_p=input_dropout
        )
        self.decoder = DecoderRNN(
            vocab_size=len(TOKENS),
            max_len=500,
            hidden_size=rnn_hidden_size,
            n_layers=rnn_num_layers,
            sos_id=TOKENS.index("SOS"),
            eos_id=TOKENS.index("EOS"),
            embedding=token_embedding
        )
        self.property_predictor = StackedDenseNet(emb_size, predictor_hidden_size, num_properties)

        self.total_rnn_hidden_dim = self.encoder.rnn.hidden_size * self.encoder.rnn.num_layers
        self.fc_enc = Linear(self.total_rnn_hidden_dim, emb_size)
        self.fc_dec = Linear(emb_size, self.total_rnn_hidden_dim)
        self.decode_function = decode_function

    def encode(self, inp, length):
        _, hidden = self.encoder(inp, length)
        hidden = torch.transpose(hidden, 0, 1).reshape(-1, self.total_rnn_hidden_dim)
        embedding = self.fc_enc(hidden)
        embedding = torch.tanh(embedding)
        """if self.emb_noise_std > 0:
            embedding = gaussian_noise(embedding, self.training, self.emb_noise_std)"""
        return embedding

    def decode(self, embedding, target_tensor=None, teacher_forcing_ratio=0, beam_search=False):
        h_0 = self.fc_dec(embedding)
        h_0 = h_0.view(-1, self.decoder.rnn.num_layers, self.decoder.rnn.hidden_size).transpose(0, 1).contiguous()
        output, _, _ = self.decoder(
            inputs=target_tensor,
            encoder_hidden=h_0,
            function=self.decode_function,
            teacher_forcing_ratio=teacher_forcing_ratio)
        output = torch.stack(output, dim=1)
        return output

    def reconstruction_criterion(self, pred, true):
        true = true[:, 1:]
        loss = F.cross_entropy(pred.flatten(end_dim=1), true.flatten(), ignore_index=PAD_VALUE)
        return loss

    def property_criterion(self, pred, true):
        loss = F.mse_loss(pred, true)
        return loss

    def forward(self, input_tensor, input_length=None, target_tensor=None, labels=None, teacher_forcing_ratio=0):
        embedding = self.encode(input_tensor, input_length)
        output = self.decode(embedding, target_tensor, teacher_forcing_ratio)
        properties = self.property_predictor(torch.squeeze(embedding))
        loss = self.reconstruction_criterion(output, target_tensor)
        loss += self.property_criterion(properties, labels)
        output = torch.argmax(output, dim=-1)
        return loss, output




