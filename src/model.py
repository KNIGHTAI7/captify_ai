import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel

class ImageCaptioningModel(nn.Module):
    """
    Image Captioning Model: ViT (encoder) + GPT-2 (decoder)
    
    Architecture:
    Image → ViT → Image Features → Projection Layer → GPT-2 → Caption
    """
    
    def __init__(self, vit_model_name='google/vit-base-patch16-224', 
             gpt2_model_name='gpt2', 
             freeze_vit=True):
        """
        Args:
            vit_model_name: Pretrained ViT model
            gpt2_model_name: Pretrained GPT-2 model
            freeze_vit: Whether to freeze ViT weights (recommended initially)
        """
        super(ImageCaptioningModel, self).__init__()
        
        # Load pretrained ViT (Vision Transformer)
        print(f"Loading ViT: {vit_model_name}")
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_size = self.vit.config.hidden_size  # 768 for base ViT
        
        # Freeze ViT weights to save memory (we'll fine-tune later if needed)
        if freeze_vit:
            print("Freezing ViT weights...")
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Load pretrained GPT-2
        print(f"Loading GPT-2: {gpt2_model_name}")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2_hidden_size = self.gpt2.config.n_embd  # 768 for base GPT-2
        
        # Resize GPT-2 embeddings to account for new special tokens
        # (We added <|pad|>, <|startoftext|>, <|endoftext|>)
        # FIXED: Removed len() since vocab_size is already an integer
        self.gpt2.resize_token_embeddings(self.gpt2.config.vocab_size + 3)
        
        # Projection layer: Maps ViT output to GPT-2 input space
        # This is the "bridge" between vision and language
        self.projection = nn.Linear(self.vit_hidden_size, self.gpt2_hidden_size)
        
        print(f"\n✓ Model Architecture:")
        print(f"  ViT hidden size: {self.vit_hidden_size}")
        print(f"  GPT-2 hidden size: {self.gpt2_hidden_size}")
        print(f"  Projection layer: {self.vit_hidden_size} → {self.gpt2_hidden_size}")
        print(f"  ViT frozen: {freeze_vit}")
    
    def forward(self, images, caption_ids, attention_mask):
        """
        Forward pass
        
        Args:
            images: [batch_size, 3, 224, 224] - preprocessed images
            caption_ids: [batch_size, seq_len] - tokenized captions
            attention_mask: [batch_size, seq_len] - attention mask
        
        Returns:
            loss: Language modeling loss
            logits: Predicted token logits
        """
        batch_size = images.size(0)
        
        # 1. Encode image with ViT
        vit_outputs = self.vit(pixel_values=images)
        
        # Get the [CLS] token representation (first token)
        # Shape: [batch_size, 768]
        image_features = vit_outputs.last_hidden_state[:, 0, :]
        
        # 2. Project image features to GPT-2 space
        # Shape: [batch_size, 768]
        image_embeddings = self.projection(image_features)
        
        # 3. Get text embeddings from GPT-2
        # Shape: [batch_size, seq_len, 768]
        text_embeddings = self.gpt2.transformer.wte(caption_ids)
        
        # 4. Prepend image embeddings to text embeddings
        # Shape: [batch_size, 1, 768] → [batch_size, 1 + seq_len, 768]
        image_embeddings = image_embeddings.unsqueeze(1)  # Add sequence dimension
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # 5. Create attention mask for combined input
        # Image token should always be attended to
        image_attention = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        combined_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        # 6. Create labels for the combined sequence
        # The image token doesn't have a label (use -100 to ignore in loss)
        # Shape: [batch_size, 1 + seq_len]
        image_labels = torch.full((batch_size, 1), -100, device=caption_ids.device, dtype=caption_ids.dtype)
        combined_labels = torch.cat([image_labels, caption_ids], dim=1)
        
        # 7. Pass through GPT-2 transformer
        outputs = self.gpt2(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels  # Now matches the sequence length!
        )
        
        return outputs.loss, outputs.logits
    
    def generate_caption(self, images, tokenizer, max_length=50, num_beams=5, temperature=1.0):
        """
        Generate captions for images using beam search
        
        Args:
            images: [batch_size, 3, 224, 224]
            tokenizer: GPT-2 tokenizer
            max_length: Maximum caption length
            num_beams: Number of beams for beam search (1=greedy, 3-5=better quality)
            temperature: Sampling temperature (lower=more conservative)
        
        Returns:
            List of generated captions
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = images.size(0)
            
            # Encode image
            vit_outputs = self.vit(pixel_values=images)
            image_features = vit_outputs.last_hidden_state[:, 0, :]
            image_embeddings = self.projection(image_features).unsqueeze(1)  # [batch, 1, 768]
            
            captions = []
            
            for i in range(batch_size):
                # Get image embedding for this sample
                single_image_emb = image_embeddings[i:i+1]  # [1, 1, 768]
                
                # Start with <|startoftext|> token
                input_ids = torch.tensor([[tokenizer.bos_token_id]], device=images.device)
                
                # Beam search setup
                beam_scores = torch.zeros(num_beams, device=images.device)
                beam_sequences = input_ids.repeat(num_beams, 1)  # [num_beams, 1]
                
                # Track completed sequences
                completed_sequences = []
                completed_scores = []
                
                for step in range(max_length):
                    # Get text embeddings for all beams
                    text_embeddings = self.gpt2.transformer.wte(beam_sequences)  # [num_beams, seq_len, 768]
                    
                    # Repeat image embedding for all beams
                    image_emb_repeated = single_image_emb.repeat(num_beams, 1, 1)  # [num_beams, 1, 768]
                    
                    # Combine
                    combined_embeddings = torch.cat([image_emb_repeated, text_embeddings], dim=1)
                    
                    # Forward pass
                    outputs = self.gpt2(inputs_embeds=combined_embeddings)
                    next_token_logits = outputs.logits[:, -1, :] / temperature  # [num_beams, vocab_size]
                    
                    # Get log probabilities
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)  # [num_beams, vocab_size]
                    
                    # Add beam scores
                    log_probs = log_probs + beam_scores.unsqueeze(1)  # [num_beams, vocab_size]
                    
                    # Flatten to find top candidates
                    vocab_size = log_probs.shape[-1]
                    log_probs_flat = log_probs.view(-1)  # [num_beams * vocab_size]
                    
                    # Get top num_beams candidates
                    top_log_probs, top_indices = torch.topk(log_probs_flat, num_beams * 2)
                    
                    # Convert flat indices back to (beam_idx, token_idx)
                    beam_indices = top_indices // vocab_size
                    token_indices = top_indices % vocab_size
                    
                    # Filter and update beams
                    new_beams = []
                    new_scores = []
                    
                    for rank, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_log_probs)):
                        # Check if this is end token
                        if token_idx == tokenizer.eos_token_id:
                            # Add to completed sequences
                            completed_seq = beam_sequences[beam_idx].tolist()
                            completed_sequences.append(completed_seq)
                            completed_scores.append(score.item())
                        else:
                            # Add to new beams
                            new_seq = torch.cat([beam_sequences[beam_idx], token_idx.unsqueeze(0)])
                            new_beams.append(new_seq)
                            new_scores.append(score.item())
                        
                        # Keep only num_beams active beams
                        if len(new_beams) >= num_beams:
                            break
                    
                    # If no active beams left, stop
                    if len(new_beams) == 0:
                        break
                    
                    # Update beams
                    max_len = max(len(seq) for seq in new_beams)
                    beam_sequences = torch.stack([
                        torch.cat([seq, torch.full((max_len - len(seq),), tokenizer.pad_token_id, device=seq.device)])
                        for seq in new_beams
                    ])
                    beam_scores = torch.tensor(new_scores, device=images.device)
                    
                    # Stop if we have enough completed sequences
                    if len(completed_sequences) >= num_beams:
                        break
                
                # Select best completed sequence
                if completed_sequences:
                    best_idx = completed_scores.index(max(completed_scores))
                    best_sequence = completed_sequences[best_idx]
                else:
                    # No completed sequence, use best active beam
                    best_sequence = beam_sequences[0].tolist()
                
                # Decode
                caption = tokenizer.decode(best_sequence, skip_special_tokens=True)
                captions.append(caption)
        
        return captions
    
    def count_parameters(self):
        """Count trainable vs frozen parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


# Test the model
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    
    print("Testing Model Architecture...\n")
    
    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {'pad_token': '<|pad|>', 'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>'}
    tokenizer.add_special_tokens(special_tokens)
    
    # Create model
    model = ImageCaptioningModel(freeze_vit=True)
    
    # Count parameters
    trainable, total = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"  Total: {total:,} ({total/1e6:.2f}M)")
    print(f"  Frozen: {total - trainable:,} ({(total-trainable)/1e6:.2f}M)")
    
    # Test forward pass with dummy data
    print(f"\nTesting forward pass...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_captions = torch.randint(0, 50257, (batch_size, 50))
    dummy_attention = torch.ones(batch_size, 50)
    
    loss, logits = model(dummy_images, dummy_captions, dummy_attention)
    
    print(f"✓ Forward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    
    # Test caption generation
    print(f"\nTesting caption generation...")
    captions = model.generate_caption(dummy_images, tokenizer, max_length=20)
    print(f"✓ Generated captions (untrained model, will be gibberish):")
    for i, cap in enumerate(captions):
        print(f"  {i+1}. {cap}")
    
    print(f"\n✓ Model is ready for training!")