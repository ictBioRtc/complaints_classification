import gradio as gr
import torch
import pandas as pd
import os
import tempfile
import time
import subprocess
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Global variables
MODEL_PATH = "local-model"
CATEGORIES = ['Online-Safety', 'BroadBand', 'TV-Radio']
idx_to_category = {0: 'Online-Safety', 1: 'BroadBand', 2: 'TV-Radio'}
TOKEN = None
TRAINING_LOGS = []
CURRENT_MODEL = None
CURRENT_TOKENIZER = None

def login_to_hf(token):
    """Login to Hugging Face"""
    global TOKEN
    TOKEN = token
    try:
        login(token)
        return "✅ Successfully logged in to Hugging Face"
    except Exception as e:
        return f"❌ Login failed: {str(e)}"

def validate_hub_model_id(username, model_name):
    """Validate and construct Hub model ID"""
    if not username or not model_name:
        return None, "Please provide both username and model name"
    
    # Clean up the model name
    model_name = model_name.strip().lower().replace(" ", "-")
    model_name = ''.join(c for c in model_name if c.isalnum() or c in ['-', '_'])
    
    # Construct the full model ID
    hub_model_id = f"{username}/{model_name}"
    
    return hub_model_id, None

def load_model(model_path):
    """Load a trained model and tokenizer"""
    global CURRENT_MODEL, CURRENT_TOKENIZER
    
    try:
        # Try loading from local path first
        if os.path.exists(model_path):
            CURRENT_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            CURRENT_MODEL = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(CATEGORIES)
            )
            return f"✅ Model loaded from local path: {model_path}"
        
        # If local path doesn't exist, try loading from Hub
        try:
            CURRENT_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            CURRENT_MODEL = BertForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(CATEGORIES)
            )
            return f"✅ Model loaded from Hugging Face Hub: {model_path}"
        except Exception as hub_error:
            # If both local and hub loading fail, fall back to base model
            CURRENT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
            CURRENT_MODEL = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(CATEGORIES)
            )
            return f"⚠️ Failed to load specified model, using base BERT model instead. Error: {str(hub_error)}"
            
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}"

def predict_text(text, model_path):
    """Make a prediction on a single text input"""
    global CURRENT_MODEL, CURRENT_TOKENIZER
    
    # Load the model if it's not loaded or a different one is requested
    if CURRENT_MODEL is None or model_path != MODEL_PATH:
        load_result = load_model(model_path)
        if load_result.startswith("❌"):
            return load_result
    
    try:
        # Tokenize input
        inputs = CURRENT_TOKENIZER(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Make prediction
        with torch.no_grad():
            outputs = CURRENT_MODEL(**inputs)
            predicted_idx = outputs.logits.argmax().item()
        
        # Get category from index
        predicted_category = idx_to_category[predicted_idx]
        
        # Check if text was truncated
        original_tokens = CURRENT_TOKENIZER(text, truncation=False)
        was_truncated = len(original_tokens['input_ids']) > 512
        truncation_warning = "\n\n⚠️ Note: This complaint was truncated to fit BERT's 512 token limit." if was_truncated else ""
        
        return f"Complaint: {text}\n\nPredicted Category: {predicted_category}{truncation_warning}"
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

def predict_csv(csv_file, model_path):
    """Make predictions on a CSV file with complaints"""
    global CURRENT_MODEL, CURRENT_TOKENIZER
    
    # Load the model if needed
    if CURRENT_MODEL is None or model_path != MODEL_PATH:
        load_result = load_model(model_path)
        if load_result.startswith("❌"):
            return load_result
    
    try:
        # Read the CSV file
        if hasattr(csv_file, 'name'):
            df = pd.read_csv(csv_file.name)
        else:
            df = pd.read_csv(csv_file)
        
        if 'complaint' not in df.columns:
            return "❌ CSV file must have a 'complaint' column"
        
        results = []
        truncated_count = 0
        
        for i, row in enumerate(df.iterrows()):
            complaint = str(row[1]['complaint'])
            
            # Check for truncation
            original_tokens = CURRENT_TOKENIZER(complaint, truncation=False)
            was_truncated = len(original_tokens['input_ids']) > 512
            if was_truncated:
                truncated_count += 1
            
            # Predict
            inputs = CURRENT_TOKENIZER(complaint, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = CURRENT_MODEL(**inputs)
                predicted_idx = outputs.logits.argmax().item()
            
            predicted_category = idx_to_category[predicted_idx]
            
            truncation_mark = " ⚠️" if was_truncated else ""
            preview = complaint if len(complaint) <= 50 else complaint[:47] + "..."
            results.append(f"Complaint {i+1}{truncation_mark}: {preview}\nPredicted Category: {predicted_category}\n")
            
            if i >= 19:
                results.append(f"... and {len(df) - 20} more (showing first 20 out of {len(df)} complaints)")
                break
        
        if truncated_count > 0:
            results.append(f"\n⚠️ {truncated_count} complaints were truncated to fit BERT's 512 token limit.")
        
        return "\n".join(results)
    except Exception as e:
        return f"❌ CSV processing failed: {str(e)}"

def train_model(dataset_name, num_epochs, batch_size, learning_rate, hf_token, 
                push_to_hub, username, model_name):
    """Start the model training process"""
    global TRAINING_LOGS, MODEL_PATH
    
    TRAINING_LOGS = []  # Reset logs at the start of training
    
    if hf_token:
        login_result = login_to_hf(hf_token)
        TRAINING_LOGS.append(login_result)
        yield "\n".join(TRAINING_LOGS)
    
    # Validate hub model ID if pushing to hub
    if push_to_hub:
        hub_model_id, error = validate_hub_model_id(username, model_name)
        if error:
            TRAINING_LOGS.append(f"❌ {error}")
            yield "\n".join(TRAINING_LOGS)
            return
    else:
        hub_model_id = None
    
    # Create training command
    cmd = [
        "python", "bert_finetune.py",
        "--dataset_name", dataset_name,
        "--model_id", "bert-base-uncased",
        "--output_dir", MODEL_PATH,
        "--feature_column", "complaint",
        "--label_column", "label_idx",
        "--num_labels", "3",
        "--num_train_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--max_length", "512"
    ]
    
    if push_to_hub and hub_model_id:
        cmd.extend(["--push_to_hub", "--hub_model_id", hub_model_id])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])
    
    TRAINING_LOGS.append(f"Starting training with command: {' '.join(cmd)}")
    yield "\n".join(TRAINING_LOGS)
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        TRAINING_LOGS.append("Training started...")
        yield "\n".join(TRAINING_LOGS)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                TRAINING_LOGS.append(line.strip())
                yield "\n".join(TRAINING_LOGS)
        
        process.wait()
        
        if process.returncode == 0:
            TRAINING_LOGS.append("✅ Training completed successfully!")
            if push_to_hub and hub_model_id:
                TRAINING_LOGS.append(f"✅ Model pushed to Hugging Face Hub: {hub_model_id}")
            
            # Load the trained model
            TRAINING_LOGS.append("Loading trained model...")
            load_result = load_model(MODEL_PATH)
            TRAINING_LOGS.append(load_result)
            
            # Final success message
            TRAINING_LOGS.append("\n✨ All done! Your model is ready to use.")
        else:
            TRAINING_LOGS.append(f"❌ Training failed with return code {process.returncode}")
        
    except Exception as e:
        TRAINING_LOGS.append(f"❌ Error during training: {str(e)}")
    
    yield "\n".join(TRAINING_LOGS)
    
def push_to_hub_after_training(model_path, username, model_name, token):
    """Push a trained model to Hugging Face Hub"""
    try:
        if not token:
            return "❌ Please provide a Hugging Face token"
        
        hub_model_id, error = validate_hub_model_id(username, model_name)
        if error:
            return f"❌ {error}"
        
        # Login and load model
        login(token)
        if not os.path.exists(model_path):
            return "❌ No trained model found. Please train a model first."
            
        try:
            model = BertForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            return f"❌ Failed to load model: {str(e)}"
            
        # Push to Hub
        try:
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            return f"✅ Successfully pushed model to {hub_model_id}"
        except Exception as e:
            return f"❌ Failed to push to Hub: {str(e)}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Create the Gradio Interface
with gr.Blocks(title="BERT Complaint Classifier") as app:
    gr.Markdown("# BERT Complaint Category Classifier")
    gr.Markdown("A simple tool to train and use a BERT model for classifying customer complaints")
    
    with gr.Tabs():
        # Training Tab
        with gr.TabItem("Train Model"):
            gr.Markdown("### Train a New Model")
            gr.Markdown("Provide your dataset information and training parameters")
            
            dataset_name = gr.Textbox(
                label="Dataset Name (from Hugging Face)", 
                placeholder="e.g., your-username/complaint-categories-dataset"
            )
            
            with gr.Row():
                num_epochs = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Epochs")
                batch_size = gr.Slider(minimum=4, maximum=32, value=8, step=4, label="Batch Size")
                learning_rate = gr.Slider(minimum=1e-5, maximum=5e-5, value=2e-5, step=1e-5, label="Learning Rate")
            
            with gr.Accordion("Hugging Face Hub Settings", open=False):
                hf_token = gr.Textbox(
                    label="Hugging Face Token (required for pushing to Hub)", 
                    type="password"
                )
                
                gr.Markdown("""### Choose when to push to Hub:
                1. During Training: Model will be pushed automatically when training completes
                2. After Training: You can push the trained model manually later""")
                
                # During Training Push
                with gr.Group():
                    push_to_hub = gr.Checkbox(
                        label="Push Model to Hub during training", 
                        value=False
                    )
                    
                    with gr.Column(visible=False) as hub_settings:
                        username = gr.Textbox(
                            label="Hugging Face Username",
                            placeholder="e.g., huggingface-username"
                        )
                        model_name = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., bert-complaint-classifier"
                        )
                
                # Post-Training Push
                with gr.Group():
                    post_train_push = gr.Checkbox(
                        label="Push trained model to Hub after training", 
                        value=False
                    )
                    
                    with gr.Column(visible=False) as post_train_settings:
                        post_train_username = gr.Textbox(
                            label="Hugging Face Username",
                            placeholder="e.g., huggingface-username"
                        )
                        post_train_model_name = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., bert-complaint-classifier"
                        )
                        post_train_token = gr.Textbox(
                            label="Hugging Face Token (if different from above)",
                            type="password"
                        )
                        post_train_push_btn = gr.Button(
                            "Push Model to Hub",
                            variant="secondary"
                        )
                        post_train_status = gr.Textbox(label="Upload Status")
                
                # Show/hide settings based on checkboxes
                push_to_hub.change(
                    lambda x: gr.update(visible=x),
                    inputs=push_to_hub,
                    outputs=hub_settings
                )
                
                post_train_push.change(
                    lambda x: gr.update(visible=x),
                    inputs=post_train_push,
                    outputs=post_train_settings
                )
            
            gr.Markdown("### BERT Model Note")
            gr.Markdown("⚠️ BERT has a maximum sequence length of 512 tokens. Complaints longer than this will be truncated.")
            
            train_btn = gr.Button("Start Training", variant="primary")
            training_output = gr.Textbox(label="Training Progress", lines=10)
            
            # Connect the buttons
            post_train_push_btn.click(
                push_to_hub_after_training,
                inputs=[
                    gr.Textbox(value=MODEL_PATH, visible=False),
                    post_train_username,
                    post_train_model_name,
                    post_train_token
                ],
                outputs=post_train_status
            )
            
            train_btn.click(
                train_model,
                inputs=[
                    dataset_name,
                    num_epochs,
                    batch_size,
                    learning_rate,
                    hf_token,
                    push_to_hub,
                    username,
                    model_name
                ],
                outputs=training_output,
                show_progress="full"  # This ensures proper progress updates
            )

        # Classification Tab
        with gr.TabItem("Classify Complaints"):
            gr.Markdown("### Classify Customer Complaints")
            
            model_path = gr.Textbox(
                label="Model Path or Hugging Face ID",
                value="local-model",
                placeholder="e.g., local-model or your-username/bert-complaint-classifier"
            )
            
            with gr.Tabs():
                # Single Complaint Classification
                with gr.TabItem("Single Complaint"):
                    text_input = gr.Textbox(
                        label="Complaint Text",
                        lines=5,
                        placeholder="Enter a customer complaint here..."
                    )
                    
                    classify_btn = gr.Button("Classify", variant="primary")
                    token_info = gr.Markdown("Note: BERT has a 512 token limit. Longer complaints will be truncated.")
                    text_output = gr.Textbox(label="Classification Result", lines=5)
                    
                    # Token counter
                    def count_tokens(text):
                        if not text or CURRENT_TOKENIZER is None:
                            return "Enter text to see token count"
                        tokens = CURRENT_TOKENIZER(text, truncation=False)
                        count = len(tokens['input_ids'])
                        if count > 512:
                            return f"⚠️ **Token count: {count}/512** - Text will be truncated for BERT"
                        else:
                            return f"Token count: {count}/512"
                    
                    text_input.change(
                        fn=count_tokens,
                        inputs=text_input,
                        outputs=token_info
                    )
                    
                    classify_btn.click(
                        predict_text,
                        inputs=[text_input, model_path],
                        outputs=text_output
                    )
                
                # Batch Processing
                with gr.TabItem("Batch Processing"):
                    gr.Markdown("Upload a CSV file with a 'complaint' column")
                    csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
                    batch_classify_btn = gr.Button("Classify All", variant="primary")
                    csv_output = gr.Textbox(label="Classification Results", lines=15)
                    
                    batch_classify_btn.click(
                        predict_csv,
                        inputs=[csv_input, model_path],
                        outputs=csv_output
                    )

# Launch the app
if __name__ == "__main__":
    if CURRENT_TOKENIZER is None:
        CURRENT_TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
    app.launch(share=True)