
import torch
import torch.nn.functional as F
from transformers import GPT2Config
from transformers import GPT2Model
from transformers import GPT2LMHeadModel
from transformers import GPT2ForSequenceClassification


class GPT(GPT2Model):
    
    def __init__(self, vocab_size, max_seqlen=1024, 
                 embed_dim=192, num_layers=6, num_heads=6,
                 eos_token_id=5, pad_token_id=5,
                 **kargs
            ):
        config = GPT2Config(
            vocab_size = vocab_size,
            n_positions = max_seqlen,
            n_embd = embed_dim,
            n_layer = num_layers,
            n_head = num_heads,
            eos_token_id = eos_token_id,
            pad_token_id = pad_token_id,
            kwargs = kargs,
        )
        super().__init__(config)
        self.max_seqlen = max_seqlen
        # resize model embedding to match new tokenizer
        self.resize_token_embeddings(vocab_size)
    

class GPTLM(GPT2LMHeadModel):

    def __init__(self, vocab_size, max_seqlen=1024, 
                 embed_dim=192, num_layers=6, num_heads=6, 
                 eos_token_id=5, pad_token_id=5,
                 **kargs
            ):
        config = GPT2Config(
            vocab_size = vocab_size,
            n_positions = max_seqlen,
            n_embd = embed_dim,
            n_layer = num_layers,
            n_head = num_heads,
            eos_token_id = eos_token_id,
            pad_token_id = pad_token_id,
            kwargs = kargs,
        )
        super().__init__(config)
        self.max_seqlen = max_seqlen
        # resize model embedding to match new tokenizer
        self.resize_token_embeddings(vocab_size)
        # fix model padding token id
        self.config.pad_token_id = self.config.eos_token_id

    @torch.no_grad()
    def next_log_probs(self, input_ids, ngram=30, temperature=1.0):
        # if the sequence context is growing too long we must crop it at ngram
        input_ids_1 = input_ids if input_ids.size(1) <= ngram else input_ids[:, -ngram:]
        # forward the model to get the logits for the index in the sequence
        logits = self(input_ids_1)[0]
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    @torch.no_grad()
    def correct(self, logits, ngram=30, min_len=5, lm_weight=0.2):
        """ correct the output transcript of a asr model with a language model
        logits: (seq_len, num_classes)
        """
        l, v = logits.size()
        assert l >= min_len, f"Minimum sequence length {min_len} is required for correction"
        assert ngram >= min_len, f"ngram {ngram} must be larger than the minimum sequence length {min_len}"
        # init input_idx with the first min_len tokens
        input_ids = torch.argmax(logits[:min_len], dim=-1).unsqueeze(0) # (1, seq_len)
        # greedy search
        for i in range(min_len, l):
            lm_probs = self.next_log_probs(input_ids, ngram=ngram).squeeze() # (num_classes,)
            raw_probs = F.log_softmax(logits[i, :], dim=-1) # (num_classes,)
            probs = (1 - lm_weight) * raw_probs + lm_weight * lm_probs[:len(raw_probs)]
            next_idx = torch.argmax(probs, dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1)
            input_ids = torch.cat((input_ids, next_idx), dim=1) # (1, seq_len)
        return input_ids
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            input_ids_1 = input_ids if input_ids.size(1) <= self.max_seqlen else input_ids[:, -self.max_seqlen:]
            # forward the model to get the logits for the index in the sequence
            logits = self(input_ids_1)[0]
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids
    

class GPTSeqClassifier(GPT2ForSequenceClassification):

    def __init__(self, num_classes, vocab_size, max_seqlen=1024, 
                 embed_dim=192, num_layers=6, num_heads=6, 
                 eos_token_id=5, pad_token_id=5,
                 **kargs
            ):
        config = GPT2Config(
            num_labels = num_classes,
            vocab_size = vocab_size,
            n_positions = max_seqlen,
            n_embd = embed_dim,
            n_layer = num_layers,
            n_head = num_heads,
            eos_token_id = eos_token_id,
            pad_token_id = pad_token_id,
            kwargs = kargs,
        )
        # super().__init__(config=config)
        super().__init__(config=config)
        self.max_seqlen = max_seqlen
        # resize model embedding to match new tokenizer
        self.resize_token_embeddings(vocab_size)
        # fix model padding token id
        self.config.pad_token_id = self.config.eos_token_id


if __name__ == '__main__':

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    model = GPTSeqClassifier(num_classes=5, vocab_size=4000, max_seqlen=256, embed_dim=384, num_layers=1, num_heads=6)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
    print(model(input_ids)[0].size())