"""

WHAT HF DOES WHEN ONLINE
1. pip install transformers
2. from transformers import pipeline
3. pipe = pipeline('text-generation', model='gpt2) # downloads ~500MB
4. pipe('To be or not to be') # generates text

OFFLINE: We'll mock the API with our CharLSTM so i learn the interface
"""

import torch
import torch.nn.functional as F

# Mock tokenizer - converts text <-> indices
class MockTokenizer:
    def __init__(self, char_to_idx, idx_to_char):
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.eos_token_id = char_to_idx.get('.', 0)

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, ids):
        return [self.idx_to_char.get(i, '?') for i in ids]
    

# Mock pipeline - same API as HF
class MockPipeline:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def __call__(self, prompt, max_length=100, temperature=0.8, do_sample=True):
        """Same signature as HF pipeline"""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        hidden = None
        output_ids = input_ids[0].tolist()

        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.model(input_ids, hidden)
                next_logits = logits[0, -1] / temperature

                if do_sample:
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                else:
                    next_id = torch.argmax(next_logits).item()

                output_ids.append(next_id)
                input_ids = torch.tensor([[next_id]]).to(self.device)

                if next_id == self.tokenizer.eos_token_id:
                    break
        
        text = self.tokenizer.decode(output_ids)
        return [{'generated_text':text}] # HF returns list of dicts
    
def load_our_model():
    """Load your trained CharLSTM from lesson 3"""
    ckpt = torch.load('saved/char_lstm.pt', map_location='cpu')
    from lesson3_char_rnn import CharLSTM
    model = CharLSTM(ckpt['vocab_size'])
    model.load_state_dict(ckpt['model_state'])
    tokenizer = MockTokenizer(ckpt['char_to_idx'], ckpt['idx_to_char'])
    return model, tokenizer

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load our trained model
    model, tokenizer = load_our_model()
    pipe = MockPipeline(model, tokenizer, device)

    # SAME API AS HUGGINGFACE!
    print('='*50)
    print('HF-STYLE GENERATION.')
    print('='*50)

    result = pipe('To be', max_length=150, temperature=0.8)
    print(result[0]['generated_text'])

    print('\n' + '='*50)
    print('WHEN YOU GET INTERNET - REAL HF CODE:')
    print('='*50)
    print("""
    from transformers import pipeline

    # Downloads GPT-2 automatically
    pipe = pipeline('text-generation', model='gpt2')

    # Same exact call!
    result = pipe('To be or not to be', max_length=100, temperature=0.8)
    print(result[0]['generated_text'])     
          """)
    
    print('\n✅ You now know HF API. Just swap MockPipeline -> pipeline() later')


"""
pipeline('text-generation') # GPT-style
pipeline('sentiment-analysis') # GERT-style
pipeline('fill-mask') # BERT: 'Paris is [MASK] of France'
pipeline('ner') # Named entity recognition
pipeline('translation_en_to_fr')
pipeline('summarization')
"""