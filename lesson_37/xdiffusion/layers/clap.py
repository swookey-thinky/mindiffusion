from msclap import CLAP
import torch
from typing import Dict, List


class FrozenCLAPTextEmbedder(torch.nn.Module):
    """Uses the CLAP transformer encoder for text (from Microsoft)"""

    def __init__(self, version="2023", device="cuda", max_length=77):
        super().__init__()
        self._clap_model = CLAP(
            version=version, use_cuda=True if device == "cuda" else False
        )
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self._clap_model.clap.caption_encoder.base = (
            self._clap_model.clap.caption_encoder.base.eval()
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, prompts: List[str], **kwargs):
        # get_text_embeddings returns the text embeddings for the class token
        # shape (B, 1024), but we can the token embeddings for all of the tokens,
        # so we need to break it apart.
        text_tokens = self.tokenize(prompts=prompts)
        text_tokens = text_tokens["input_ids"]
        outputs = self._clap_model.clap.caption_encoder.base(input_ids=text_tokens)
        z = self._clap_model.clap.caption_encoder.projection(outputs.last_hidden_state)
        return z

    def encode(self, prompts: List[str]):
        """Gets image and text embeddings using CLIP.

        Args:
            images: Tensor batch of unnormalized image data
            prompts: List of string prompts.

        Returns:
            Tuple of:
                image_embeddings: Tensor batch of image embeddings
                text_embeddings: Tensor batch of prompt embeddings
                text_encodings: Tensor batch of prompt encodings
        """
        return self(prompts)

    def tokenize(self, prompts: List[str]):
        tokenized_texts = []
        for ttext in prompts:
            if "gpt" in self._clap_model.args.text_model:
                ttext = ttext + " <|endoftext|>"
            tok = self._clap_model.tokenizer.encode_plus(
                text=ttext,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            for key in self._clap_model.token_keys:
                tok[key] = (
                    tok[key].reshape(-1).cuda()
                    if self._clap_model.use_cuda and torch.cuda.is_available()
                    else tok[key].reshape(-1)
                )
            tokenized_texts.append(tok)
        return self._clap_model.default_collate(tokenized_texts)
