
from __future__ import unicode_literals, print_function, division  # Enable Python 2/3 compatibility features

from io import open  # Import open() with Unicode support
import unicodedata  # Unicode character processing utilities
import re  # Standard regular expression module
import random  # Random number generation

import torch  # Core PyTorch library
import sentencepiece as spm
import torch.nn as nn  # Neural network modules
from torch import optim  # Optimization algorithms
import torch.nn.functional as F  # Functional API (losses, activations)

import numpy as np  # Numerical computing library
from torch.utils.data import TensorDataset, DataLoader, RandomSampler  # Dataset and dataloader utilities
from torch.utils.data import random_split  # For splitting datasets into training and validation subsets

import regex as re  # Advanced regex library (overrides standard re)
import unicodedata  # Unicode normalization and character info (re-imported)

import time  # Time measurement utilities
import math  # Mathematical functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU

# ----------------------- PLOT UTILS -------------------

import matplotlib.pyplot as plt  # Plotting library
plt.switch_backend('agg')  # Use non-interactive backend (useful for saving plots without GUI)
import matplotlib.ticker as ticker  # Tools for controlling axis tick locations
import numpy as np  # Numerical computing library

import matplotlib.pyplot as plt
from matplotlib import ticker

def plot_values(values, y_tick_interval=0.2, title="Plot", xlabel="Step", ylabel="Value"):

    fig, ax = plt.subplots()
    
    # Set y-axis ticks at regular intervals
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=y_tick_interval))
    
    # Plot the values
    ax.plot(values, marker='o', linestyle='-')  # optional: markers for better visibility
    
    # Labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.show()

# ----------------------- TIME UTILS -----------------------

def format_time(seconds: float) -> str:
    minutes = math.floor(seconds / 60)
    seconds_remaining = int(seconds - minutes * 60)
    return f"{minutes} m {seconds_remaining} sec"

def elapsed_time(start_time: float, progress: float) -> str:
    now = time.time()
    elapsed = now - start_time
    estimated_total = elapsed / progress
    remaining = estimated_total - elapsed
    return f"{format_time(elapsed)} remaining: {format_time(remaining)}"

# ------------------------- READING AND PREPROCESSING DATA -----------------

sos = 0  # Start-Of-Sentence token index
eos = 1  # End-Of-Sentence token index

class lang_voc:  # Language vocabulary class

    def __init__(self, name):  # Constructor
        self.name = name  # Name of the language
        self.idx_to_w = {0: "SOS", 1: "EOS"}  # Reverse mapping from indices to words
        self.n_words = 2  # Total word count (including SOS and EOS)
        self.w_to_idx = {}  # Dictionary mapping words to indices
        self.w_to_count = {}  # Dictionary counting word frequencies

        # Add a special unknown token so that unseen subwords or words can be mapped safely. 
        unk_token = '<UNK>'
        self.w_to_idx[unk_token] = self.n_words
        self.w_to_count[unk_token] = 0
        self.idx_to_w[self.n_words] = unk_token
        self.n_words += 1

        self.sp_model = None # SentencePiece model for subword tokenization

    def add_sentence(self, sentence):  # Add all words from a sentence to vocabulary
        for word in sentence.split(' '):  # Split sentence into words
            self.add_w(word)  # Add each word individually

    def add_w(self, word):  # Add a single word to vocabulary
        if word not in self.w_to_idx:  # If word is new
            self.w_to_idx[word] = self.n_words  # Assign a new index
            self.w_to_count[word] = 1  # Initialize word count
            self.idx_to_w[self.n_words] = word  # Map index back to word
            self.n_words += 1  # Increment total word count
        else:
            self.w_to_count[word] += 1  # Increment count if word already exists

class Normalizer:  # Text normalization class

    def __init__(self, type='word'):  # Constructor with normalization type
        self.type = type  # Store normalization type

    def normalize_w(self, s):  # Function to normalize and clean a text string
        s = s.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
        s = unicodedata.normalize("NFC", s)  # Normalize Unicode characters (canonical composition)

        # Space out punctuation
        s = re.sub(r"([.!?])", r" \1", s)  # Add space before punctuation marks

        # Keep ALL Unicode letters + ! ? .
        s = re.sub(r"[^\p{L}!?\.]+", " ", s)  # Replace non-letter and non-punctuation chars with space

        return s.strip()  # Remove extra spaces and return cleaned string

    def normalize_logic(self, s):
        
        s = unicodedata.normalize("NFC", s.strip().lower()) # Unicode normalization

        s = re.sub(r"([.!?;])", r" \1", s) # Space out punctuation (keep ! ? . ; ->)

        # Standardize logical operators spacing
        s = re.sub(r"\bnot\b", "NOT", s, flags=re.IGNORECASE) # Replace 'not' with 'NOT'
        s = re.sub(r"\band\b", "AND", s, flags=re.IGNORECASE) # Replace 'and' with 'AND'
        s = re.sub(r"\bor\b", "OR", s, flags=re.IGNORECASE) # Replace 'or' with 'OR'
        s = re.sub(r"->", "->", s)  # Keep implication
        s = re.sub(r";", " ; ", s)  # Ensure spacing around semicolons

        # Remove any unwanted characters, keep letters, numbers, and logical symbols
        s = re.sub(r"[^A-Za-z0-9\s!?.;->]+", " ", s)

        s = re.sub(r"\s+", " ", s) # Collapse multiple spaces

        return s.strip()
    
    def normalize(self, s): # General normalization function that delegates to specific methods based on type
        if self.type == 'word':
            return self.normalize_w(s)
        elif self.type == 'logic':
            return self.normalize_logic(s)
        else:
            raise ValueError("Unsupported normalization type")

def init_lang(lang1, lang2):  # Function to read and preprocess bilingual sentence pairs

    # Read the file and split into lines.  Prefer to load from the
    # `data/` directory, but fall back to the current directory if the
    # expected file does not exist. 
    file_path = f'data/{lang1}-{lang2}.txt'
    try:
        with open(file_path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
    except FileNotFoundError:
        alt_path = f'{lang1}-{lang2}.txt'
        with open(alt_path, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')

    if lang1 == 'logic' or lang2 == 'logic': # If either language is 'logic', use the logic normalizer
        normalizer = Normalizer(type='logic').normalize
    else:
        normalizer = Normalizer(type='word').normalize

    # Split every line into pairs and normalize
    pairs = [[normalizer(s) for s in l.split('\t')] for l in lines]  

    in_lang = lang_voc(lang1)   # Create input language vocabulary object
    out_lang = lang_voc(lang2)  # Create output language vocabulary object

    return in_lang, out_lang, pairs  # Return language objects and sentence pairs

def filter_pairs(pairs, max_length):
        
    return [
        pair for pair in pairs
        if len(pair[0].split(' ')) < max_length  # Check input length
        and len(pair[1].split(' ')) < max_length  # Check output length
    ]

def read_pair(lang1, lang2, max_length):  # Function to load, filter, and prepare language data
    
    in_lang, out_lang, pairs = init_lang(lang1, lang2)  # Read raw sentence pairs and initialize languages
    print("%s sentence pairs readed" % len(pairs))  # Display number of sentence pairs read

    pairs = filter_pairs(pairs, max_length)  # Filter sentence pairs based on length and prefixes

    print("reduced to %s sentence pairs" % len(pairs))  # Display number after filtering

    for pair in pairs:  # Iterate through filtered sentence pairs

        in_lang.add_sentence(pair[0])  # Add input sentence words to input language vocabulary
        out_lang.add_sentence(pair[1])  # Add output sentence words to output language vocabulary

    print(in_lang.name, in_lang.n_words)  # Print input language name and vocabulary size
    print(out_lang.name, out_lang.n_words)  # Print output language name and vocabulary size

    return in_lang, out_lang, pairs  # Return prepared language objects and filtered sentence pairs

# ------------------------------ GRU UNIT ----------------------------

class GRU_unit(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.input_size = input_size # Input feature size
        self.hidden_size = hidden_size # Hidden state size
        self.batch_first = batch_first # Whether batch dimension is first

        self.W_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size)) # Weights for input to hidden
        self.W_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size)) # Weights for hidden to hidden

        self.b_ih = nn.Parameter(torch.zeros(3 * hidden_size)) # Bias for input to hidden
        self.b_hh = nn.Parameter(torch.zeros(3 * hidden_size)) # Bias for hidden to hidden

        self.reset_parameters() # Initialize weights

    def reset_parameters(self):
        for p in self.parameters(): # Iterate over all parameters
            if p.dim() > 1: # Only initialize weight matrices
                nn.init.xavier_uniform_(p) # Xavier uniform initialization

    def forward(self, x, h0=None):

        if not self.batch_first: 
            x = x.transpose(0, 1)  # -> (batch, seq_len, input_size)

        batch_size, seq_len, _ = x.shape # Get batch size and sequence length

        if h0 is None: # If no initial hidden state is provided
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device) # Initialize hidden state to zeros
        else:
            h_t = h0.squeeze(0) # Remove extra dimension

        outputs = [] # Store outputs for each timestep

        for t in range(seq_len): # Loop over each timestep
            x_t = x[:, t, :]  # (batch, input_size)

            gates = ( 
                torch.matmul(x_t, self.W_ih.T) + self.b_ih + # Input contribution
                torch.matmul(h_t, self.W_hh.T) + self.b_hh # Hidden state contribution
            )

            z_t, r_t, n_t = gates.chunk(3, dim=1) # Split gates into update, reset, and new gates

            z_t = torch.sigmoid(z_t) # Update gate
            r_t = torch.sigmoid(r_t) # Reset gate
            n_t = torch.tanh(n_t + r_t * (h_t @ self.W_hh[self.hidden_size*2:].T)) # New gate

            h_t = (1 - z_t) * h_t + z_t * n_t # Update hidden state
            outputs.append(h_t.unsqueeze(1)) # Store output

        output = torch.cat(outputs, dim=1) # Concatenate outputs along sequence dimension
        h_n = h_t.unsqueeze(0) # Final hidden state with added dimension

        if not self.batch_first: # If original input was not batch first
            output = output.transpose(0, 1) # -> (seq_len, batch, hidden_size)

        return output, h_n # Return all outputs and final hidden state

# ------------------------------ ENCODER AND DECODER NETWORKS ----------------------------

class rnn_encoder(nn.Module):  # Encoder RNN class inheriting from PyTorch's nn.Module
    
    def __init__(self, input_size, h_size, dropout_p=0.01):  # Constructor with vocabulary and hidden sizes

        super(rnn_encoder, self).__init__()  # Initialize the parent nn.Module
        self.h_size = h_size  # Store hidden state size

        self.embedding = nn.Embedding(input_size, h_size)  # Convert word indices to dense vectors
        self.gru = GRU_unit(h_size, h_size, batch_first=True)  # GRU layer for sequence processing
        self.dropout = nn.Dropout(dropout_p)  # Dropout layer to prevent overfitting

    def forward(self, input):  # Forward pass of the encoder

        embedded = self.dropout(self.embedding(input))  # Embed input tokens and apply dropout
        output, hidden = self.gru(embedded)  # Pass embeddings through GRU
        return output, hidden  # Return all outputs and final hidden state

class rnn_att_decoder(nn.Module):

    def __init__(self, h_size, out_size, dropout_p=0.01, max_length=10, n_heads=1, n_layers=1):

        super(rnn_att_decoder, self).__init__()

        if h_size % n_heads != 0:
            raise ValueError("hidden size must be divisible by the number of heads")
        
        self.h_size = h_size # Hidden state size
        self.max_length = max_length # Maximum output sequence length
        self.n_heads = n_heads # Number of attention heads
        self.head_dim = h_size // n_heads # Dimension of each attention head
        self.n_layers = n_layers # Number of GRU layers in the decoder

        # Embedding layer
        self.embedding = nn.Embedding(out_size, h_size)
        self.dropout = nn.Dropout(dropout_p)

        # Linear projections for queries, keys and values and output projection
        self.W_Q = nn.Linear(h_size, h_size, bias=False)
        self.W_K = nn.Linear(h_size, h_size, bias=False)
        self.W_V = nn.Linear(h_size, h_size, bias=False)
        self.W_O = nn.Linear(h_size, h_size, bias=False)

        # GRU layers: first layer expects concatenated [embedding; attention]
        gru_layers = []

        for i in range(n_layers):
            if i == 0:
                gru_layers.append(GRU_unit(2 * h_size, h_size, batch_first=True))
        
            else:
                gru_layers.append(GRU_unit(h_size, h_size, batch_first=True))
        
        self.gru_layers = nn.ModuleList(gru_layers)
        
        # Final projection to output vocabulary
        self.out = nn.Linear(h_size, out_size)

    def forward(self, encoder_out, encorder_h, tens_final=None):
    
        # Determine batch size
        batch_size = encorder_h.size(1)

        # Initialise decoder hidden state for each layer by repeating encoder hidden
        # encorder_h shape: (1, batch, h_size) -> expand to (n_layers, batch, h_size)
        decoder_h = encorder_h.repeat(self.n_layers, 1, 1)
        
        # Initial input token: SOS for each sample
        decoder_in = torch.empty(batch_size, 1, dtype=torch.long, device=encorder_h.device).fill_(sos)
        
        # Memory for attention: start with the final hidden layer state.  
        # Use slicing to preserve the layer dimension so that permute works
        memory = decoder_h[-1:].permute(1, 0, 2).contiguous()  # (batch, 1, h_size)

        # Collect outputs and attention weights
        outputs = []
        attn_weights_per_step = []
        
        # Determine number of steps
        max_steps = tens_final.size(1) if tens_final is not None else self.max_length
        
        for i in range(max_steps):
        
            # Perform one decoding step
            step_out, decoder_h, attn_w, memory = self.forward_step(decoder_in, decoder_h, memory)
            outputs.append(step_out)
            attn_weights_per_step.append(attn_w)
        
            # Teacher forcing or greedy decoding for next input
            if tens_final is not None:
        
                if i >= tens_final.size(1):
                    break
                decoder_in = tens_final[:, i].unsqueeze(1)
        
            else:
                # Greedy: take argmax
                _, topi = step_out.topk(1)
                decoder_in = topi.squeeze(-1).detach()
        
        # Concatenate outputs along time dimension and apply log_softmax
        if outputs:
            decoder_out = torch.cat(outputs, dim=1)
            decoder_out = F.log_softmax(decoder_out, dim=-1)
        else:
            decoder_out = torch.empty(batch_size, 0, self.out.out_features, device=encorder_h.device)
        
        # Pad attention weights to a uniform length
        if attn_weights_per_step:
            final_mem_len = memory.size(1)
            dec_steps = len(attn_weights_per_step)
            attn_padded = torch.zeros(batch_size, dec_steps, final_mem_len, device=encorder_h.device)
            for t, w in enumerate(attn_weights_per_step):
                mem_len_t = w.size(-1)
                attn_padded[:, t, :mem_len_t] = w.squeeze(1)
        
        else:
            attn_padded = torch.empty(batch_size, 0, memory.size(1), device=encorder_h.device)
        
        return decoder_out, decoder_h, attn_padded

    def forward_step(self, decoder_in, hidden, memory):

        batch_size = decoder_in.size(0)
        # Embed input token and apply dropout
        embedded = self.dropout(self.embedding(decoder_in))  # (batch, 1, h_size)
        # Compute queries, keys and values for multi‑head attention
        q = self.W_Q(embedded)  # (batch, 1, h_size)
        k = self.W_K(memory)    # (batch, mem_len, h_size)
        v = self.W_V(memory)    # (batch, mem_len, h_size)
        # Reshape for multi‑head attention
        # q: (batch, n_heads, 1, head_dim)
        q = q.view(batch_size, 1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # k, v: (batch, n_heads, mem_len, head_dim)
        mem_len = memory.size(1)
        k = k.view(batch_size, mem_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, mem_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Scaled dot‑product attention
        # scores: (batch, n_heads, 1, mem_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        # Compute context: (batch, n_heads, 1, head_dim)
        context = torch.matmul(attn_weights, v)
        # Concatenate heads: (batch, 1, h_size)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, 1, self.n_heads * self.head_dim)
        # Linear projection of context
        attn_output = self.W_O(context)  # (batch, 1, h_size)
        # Process through stacked GRU layers
        new_hidden_states = []
        layer_input = None
        for i, gru in enumerate(self.gru_layers):
            if i == 0:
                # Concatenate embedding and attention output for first layer
                gru_input = torch.cat((embedded, attn_output), dim=2)
            else:
                gru_input = layer_input
            # Extract previous hidden for this layer
            h0 = hidden[i].unsqueeze(0)  # (1, batch, h_size)
            # Compute GRU output
            out, h_next = gru(gru_input, h0)
            # out: (batch, 1, h_size)
            new_hidden_states.append(h_next)
            # Input for next layer is the output of current layer
            layer_input = out
        # Stack new hidden states: list of tensors with shape (1,batch,h_size)
        new_hidden = torch.cat(new_hidden_states, dim=0)  # (n_layers, batch, h_size)
        # Append final layer hidden state to memory.  Use slicing to retain
        # three dimensions so that permute works correctly when n_layers=1
        memory = torch.cat((memory, new_hidden[-1:].permute(1, 0, 2)), dim=1)
        # Compute logits from last layer output
        logits = self.out(layer_input)  # (batch, 1, out_size)
        # Average attention weights across heads for interpretability.  Reduce
        # the head dimension without retaining it so that the resulting
        # tensor has shape (batch, 1, mem_len).  This avoids extra
        # dimensions that would break downstream padding logic.
        attn_avg = attn_weights.mean(dim=1)  # (batch, 1, mem_len)
        return logits, new_hidden, attn_avg, memory

# -------------------------------- DATALOADER --------------------------- 

def phrase_to_idx(lang, sentence):  # Convert a sentence or BPE token sequence to indices
 
    # If a SentencePiece model is available, delegate encoding to it.
    if getattr(lang, 'sp_model', None) is not None:
        # The SentencePiece processor returns a list of ints.  We disable
        # automatic addition of BOS/EOS here; EOS will be appended
        # separately when constructing tensors.
        try:
            return lang.sp_model.encode(sentence, out_type=int)
        except TypeError:
            # For older versions of sentencepiece that require positional
            # arguments, fall back to standard encoding.
            return lang.sp_model.encode(sentence)
    # Fall back to dictionary lookup for pre‑tokenised sentences
    unk_idx = lang.w_to_idx.get('<UNK>')
    return [lang.w_to_idx.get(token, unk_idx) for token in sentence.split(' ')]

def phrase_to_tensor(lang, sentence):  # Convert a sentence into a PyTorch tensor

    indexes = phrase_to_idx(lang, sentence)  # Get word indices
    indexes.append(eos)  # Add End-Of-Sentence token
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)  # Return tensor shaped [1, seq_len]

def get_train_val_dataloaders(batch_size, lang1, lang2, max_length, valid_ratio=0.1, num_merges=6000):

    # Read and filter sentence pairs
    in_lang_tmp, out_lang_tmp, pairs = read_pair(lang1, lang2, max_length)
    # Extract sentences from the pairs
    src_sentences = [p[0] for p in pairs]
    tgt_sentences = [p[1] for p in pairs]
    # If num_merges is specified as 'auto' (string) or a negative value, choose
    # the best merge count by evaluating simple corpus statistics.  The
    # objective balances subword sequence length against vocabulary size.
    # Determine whether to use SentencePiece tokeniser
    # If num_merges is given as a tuple (or we add a new flag), we can detect sp.  For backward
    # compatibility we retain the old behaviour when sentencepiece is not available.
    use_sp = False
    # If num_merges is a tuple of (vocab_size, 'sp'), we interpret as using sentencepiece
    if isinstance(num_merges, (tuple, list)) and len(num_merges) >= 2:
        if 'sp' in num_merges or 'sentencepiece' in num_merges:
            use_sp = True
            # Use the first element as vocabulary size if numeric, else default
            if isinstance(num_merges[0], int):
                sp_vocab_size = num_merges[0]
            else:
                sp_vocab_size = 6000
        else:
            sp_vocab_size = None
    else:
        sp_vocab_size = None
    # If spm is available and user selected sp, enable sp usage
    if use_sp and spm is None:
        print("Warning: sentencepiece library not installed; falling back to scratch BPE.")
        use_sp = False

    # If using SentencePiece, train tokenizer models and build dataset
    if use_sp:
        # Ensure SentencePiece library is available
        if spm is None:
            raise RuntimeError("SentencePiece requested but library not available")
        # Determine vocabulary size
        # Default vocabulary size when unspecified is 8000 tokens
        vocab_size = sp_vocab_size if sp_vocab_size is not None else 8000
        # Write training corpora to temporary files
        import tempfile, os
        # Source language corpus file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as src_tmp:
            src_tmp_path = src_tmp.name
            for s in src_sentences:
                src_tmp.write(s + '\n')
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tgt_tmp:
            tgt_tmp_path = tgt_tmp.name
            for s in tgt_sentences:
                tgt_tmp.write(s + '\n')
        # Define model prefixes in a temporary directory
        src_prefix = src_tmp_path + '_spm'
        tgt_prefix = tgt_tmp_path + '_spm'
        # Train SentencePiece models for source and target languages
        # Use BPE model type to mimic Byte Pair Encoding
        spm.SentencePieceTrainer.train(
            input=src_tmp_path,
            model_prefix=src_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            bos_id=0,
            eos_id=1,
            unk_id=2,
            pad_id=3
        )
        spm.SentencePieceTrainer.train(
            input=tgt_tmp_path,
            model_prefix=tgt_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=1.0,
            bos_id=0,
            eos_id=1,
            unk_id=2,
            pad_id=3
        )
        # Load the trained models
        sp_src = spm.SentencePieceProcessor()
        sp_src.load(src_prefix + '.model')
        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.load(tgt_prefix + '.model')
        # Create vocabulary objects and attach the SentencePiece processors
        in_lang = lang_voc(lang1)
        out_lang = lang_voc(lang2)
        in_lang.sp_model = sp_src
        out_lang.sp_model = sp_tgt
        # Determine vocabulary sizes
        in_lang.n_words = sp_src.get_piece_size()
        out_lang.n_words = sp_tgt.get_piece_size()
        # Encode all sentences and build padded matrices
        n = len(src_sentences)
        # Encode sentences and record max length
        encoded_src = []
        encoded_tgt = []
        max_len = 0
        for s in src_sentences:
            # SentencePiece encode returns a list of ints without BOS/EOS by default
            ids = sp_src.encode(s, out_type=int)
            # Append EOS token explicitly
            ids.append(sp_src.eos_id())
            encoded_src.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)
        for s in tgt_sentences:
            ids = sp_tgt.encode(s, out_type=int)
            ids.append(sp_tgt.eos_id())
            encoded_tgt.append(ids)
            if len(ids) > max_len:
                max_len = len(ids)
        # Allocate arrays filled with pad token (PAD id = 3)
        input_ids = np.full((n, max_len), fill_value=sp_src.pad_id(), dtype=np.int32)
        target_ids = np.full((n, max_len), fill_value=sp_tgt.pad_id(), dtype=np.int32)
        for idx, (src_ids, tgt_ids) in enumerate(zip(encoded_src, encoded_tgt)):
            input_ids[idx, :len(src_ids)] = src_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids
        # Create TensorDataset and DataLoaders
        dataset = TensorDataset(torch.LongTensor(input_ids).to(device),
                                torch.LongTensor(target_ids).to(device))
        # Split into training and validation subsets
        val_size = int(len(dataset) * valid_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=RandomSampler(val_dataset))
        # Store maximum length on language objects
        in_lang.max_len = max_len
        out_lang.max_len = max_len
        # Print vocabulary sizes
        print(f"{in_lang.name} vocab size after SentencePiece: {in_lang.n_words}")
        print(f"{out_lang.name} vocab size after SentencePiece: {out_lang.n_words}")
        # Remove temporary files
        os.remove(src_tmp_path)
        os.remove(tgt_tmp_path)
        # Remove model and vocab files
        os.remove(src_prefix + '.model')
        os.remove(src_prefix + '.vocab')
        os.remove(tgt_prefix + '.model')
        os.remove(tgt_prefix + '.vocab')
        return in_lang, out_lang, train_loader, val_loader, max_len

# ------------------------------ TRAINING -----------------------------------

def single_epoch(dataloader, encoder, decoder, enc_opt,
                dec_opt, loss_type):
    
    tot = 0  # Initialize total loss for this epoch

    for data in dataloader:  # Loop over each batch
        in_tens, tens_target = data  # Unpack input and target sequences

        enc_opt.zero_grad()  # Clear encoder gradients
        dec_opt.zero_grad()  # Clear decoder gradients

        enc_out, enc_hid = encoder(in_tens)  # Encode input sequence

        dec_out, _, _ = decoder(enc_out, enc_hid, tens_target)  # Decode using encoder output

        loss = loss_type(
            dec_out.view(-1, dec_out.size(-1)),  # Flatten predictions for loss
            tens_target.view(-1)  # Flatten targets to match predictions
        )

        loss.backward()  # Backpropagate the loss

        enc_opt.step()  # Update encoder parameters
        dec_opt.step()  # Update decoder parameters

        tot += loss.item()  # Accumulate batch loss

    return tot / len(dataloader)  # Return average loss for the epoch


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
          print_step=50, plot_step=50, valid_dataloader=None):
  
    start = time.time()
    # Lists to record per‑epoch training and validation losses
    train_losses = []
    val_losses = []
    # Initialise optimisers and loss function
    enc_opt = optim.Adam(encoder.parameters(), lr=learning_rate)
    dec_opt = optim.Adam(decoder.parameters(), lr=learning_rate)
    loss_type = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):

        # Training loss for this epoch
        train_loss = single_epoch(train_dataloader, encoder, decoder, enc_opt, dec_opt, loss_type)
        train_losses.append(train_loss)
        
        # Compute validation loss if available
        val_loss = None
        
        if valid_dataloader is not None:
            with torch.no_grad():
                total = 0.0
                for data in valid_dataloader:
                    in_tens, tens_target = data
                    enc_out, enc_hid = encoder(in_tens)
                    dec_out, _, _ = decoder(enc_out, enc_hid, tens_target)
                    loss_val = loss_type(dec_out.view(-1, dec_out.size(-1)), tens_target.view(-1))
                    total += loss_val.item()
                val_loss = total / len(valid_dataloader)
            val_losses.append(val_loss)
        
        # Print progress at specified intervals
        if epoch % print_step == 0:
            if valid_dataloader is not None and val_loss is not None:
                print('%s (%d %d%%) train: %.4f val: %.4f' %
                      (elapsed_time(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, train_loss, val_loss))
            else:
                print('%s (%d %d%%) train: %.4f' %
                      (elapsed_time(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, train_loss))
    
    # Plot the recorded loss curves
    if valid_dataloader is not None:
        plot_values(train_losses, title="Training Loss", xlabel="Epoch", ylabel="Loss")
        plot_values(val_losses, title="Validation Loss", xlabel="Epoch", ylabel="Loss")
    else:
        plot_values(train_losses, title="Training Loss", xlabel="Epoch", ylabel="Loss")

# ------------------------------ EVALUATION ----------------------------

def evaluate(encoder, decoder, phrase, in_lang, out_lang):  # Evaluate a single sentence
   
    with torch.no_grad():
        # If the input language has a SentencePiece model attached, the
        # phrase is passed through as raw text; the encoding happens
        # inside `phrase_to_tensor` via the attached processor.  When
        # SentencePiece is not used, `phrase_to_tensor` will fall back
        # to dictionary lookup based on whitespace‑split tokens.
        if getattr(in_lang, 'sp_model', None) is not None:
            encoded_phrase = phrase
        else:
            encoded_phrase = phrase
        # Convert the input phrase to a tensor and run it through the encoder
        in_tens = phrase_to_tensor(in_lang, encoded_phrase)
        enc_out, encorder_h = encoder(in_tens)
        # Run the decoder without teacher forcing.  `dec_out` has shape
        # (batch=1, seq_len, vocab_size) containing log probabilities.
        dec_out, _, decoder_attn = decoder(enc_out, encorder_h)
        # Select the most likely token at each timestep
        _, topi = dec_out.topk(1)
        # Flatten the predictions to a one‑dimensional tensor
        dec_idx = topi.squeeze()
        # Build the list of output tokens
        # If using SentencePiece on the output side, decode the full
        # sequence of ids (up to EOS) via the model.  Otherwise use
        # `idx_to_w` with a safe fallback to '<UNK>' for unknown ids.
        if getattr(out_lang, 'sp_model', None) is not None:
            # Convert tensor elements to Python ints
            ids = [int(i.item()) for i in dec_idx]
            # Remove tokens after EOS (inclusive).  Use the SentencePiece
            # library to determine the EOS id when available; fall back to
            # the global EOS constant otherwise.
            try:
                eos_id = out_lang.sp_model.eos_id()
            except AttributeError:
                eos_id = eos
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            # Decode into a string and split into words
            try:
                decoded_text = out_lang.sp_model.decode(ids)
            except AttributeError:
                # For older versions of sentencepiece that expect positional
                # arguments
                decoded_text = out_lang.sp_model.decode(ids)
            dec_w = decoded_text.strip().split()
        else:
            # Not using SentencePiece: map each predicted id to its word
            # representation.  Unknown ids fall back to the special
            # unknown token when present.  Stop at the EOS token.
            dec_w_list = []
            for idx in dec_idx:
                token_id = int(idx.item())
                if token_id == eos:
                    break
                # Retrieve the word corresponding to this id, or '<UNK>'
                token = out_lang.idx_to_w.get(token_id, '<UNK>')
                dec_w_list.append(token)
            dec_w = dec_w_list
    return dec_w, decoder_attn

# ----------------------------- ATTENTION PLOTTING ----------------------------

def show_attention(in_prhase, out_w, attn):  # Visualize attention weights

    fig = plt.figure()  # Create a new figure
    ax = fig.add_subplot(111)  # Add a subplot
    cax = ax.matshow(attn.cpu().numpy(), cmap='plasma')  # Display attention matrix as an image

    # Set up axes labels
    ax.set_xticklabels([''] + in_prhase.split(' ') + ['<EOS>'], rotation=90)  # Input words on x-axis
    ax.set_yticklabels([''] + out_w)  # Output words on y-axis

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # Tick for each input word
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # Tick for each output word

    plt.show()  # Display the plot

def evaluate_and_show_attention(in_prhase, encoder, decoder, in_lang, out_lang):  # Evaluate and plot attention
    out_w, attn = evaluate(encoder, decoder, in_prhase, in_lang, out_lang)  # Get model output and attention
    print('input =', in_prhase)  # Print input sentence
    print('output =', ' '.join(out_w))  # Print decoded output sentence
    show_attention(in_prhase, out_w, attn[0, :len(out_w), :])  # Plot attention for current sequence
