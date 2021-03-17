import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from models.resnet import resnet18


class FusionNetwork(nn.Module):
    def __init__(self, args, n_output):
        super().__init__()
        self.args = args
        self.resnet = resnet18(pretrained=True, progress=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        with torch.no_grad():
            self.cls_emb = nn.Parameter(self.bert.embeddings.word_embeddings(torch.tensor(tokenizer.cls_token_id)))

        self.fc = nn.Linear(self.bert.config.hidden_size, n_output)
        self.project = nn.Linear(512, self.bert.config.hidden_size)
        self.conv = nn.Conv3d(512, self.bert.config.hidden_size, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.zeros(*x.shape[:2], 512, 7, 7).to(self.args.device)
        for i in range(x.shape[1]):
            # encode every slices
            features[:, i] = self.resnet.feature(x[:, i])

        # fusion: B * T * D * H * W
        # x = self.conv(x)
        x = self.pool(features.view(-1, *features.shape[2:]))
        x = x.view(*features.shape[:3])
        x = self.project(x)
        x = torch.cat([self.cls_emb[None, None, :].repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.bert(inputs_embeds=x).pooler_output
        return self.fc(x)
