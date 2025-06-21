#!/usr/bin/env python3
"""
BERT Fine-tuning Script for Complaint Classification
Supports training on Hugging Face datasets and pushing to Hub
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from huggingface_hub import login

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune BERT for complaint classification")
    
    # Required arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset on Hugging Face Hub")
    parser.add_argument("--model_id", type=str, default="bert-base-uncased",
                        help="Pre-trained model identifier")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the trained model")
    
    # Dataset configuration
    parser.add_argument("--feature_column", type=str, default="complaint",
                        help="Name of the text feature column")
    parser.add_argument("--label_column", type=str, default="label_idx",
                        help="Name of the label column")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of classification labels")
    
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    
    # Hugging Face Hub settings
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub after training")
    parser.add_argument("--hub_model_id", type=str,
                        help="Model ID for Hugging Face Hub")
    parser.add_argument("--hf_token", type=str,
                        help="Hugging Face authentication token")
    
    # Additional settings
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation steps during training")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log training progress every N steps")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Early stopping patience")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_and_prepare_dataset(dataset_name, feature_column, label_column):
    """Load and prepare the dataset"""
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name)
        logger.info(f"Dataset loaded successfully")
        logger.info(f"Dataset structure: {dataset}")
        
        # Check if dataset has train/test splits
        if 'train' not in dataset:
            logger.error("Dataset must have a 'train' split")
            sys.exit(1)
        
        # Verify required columns exist
        train_features = dataset['train'].features
        if feature_column not in train_features:
            logger.error(f"Feature column '{feature_column}' not found in dataset")
            logger.info(f"Available columns: {list(train_features.keys())}")
            sys.exit(1)
        
        if label_column not in train_features:
            logger.error(f"Label column '{label_column}' not found in dataset")
            logger.info(f"Available columns: {list(train_features.keys())}")
            sys.exit(1)
        
        # Create validation split if it doesn't exist
        if 'validation' not in dataset:
            logger.info("No validation split found, creating one from train split")
            dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
            dataset = DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test']
            })
        
        # Log dataset statistics
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Validation samples: {len(dataset['validation'])}")
        
        # Show label distribution
        train_labels = dataset['train'][label_column]
        unique_labels = set(train_labels)
        logger.info(f"Unique labels: {sorted(unique_labels)}")
        
        for label in sorted(unique_labels):
            count = train_labels.count(label)
            logger.info(f"Label {label}: {count} samples ({count/len(train_labels)*100:.1f}%)")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        sys.exit(1)

def tokenize_function(examples, tokenizer, feature_column, max_length):
    """Tokenize the input text"""
    return tokenizer(
        examples[feature_column],
        truncation=True,
        padding=False,  # Padding will be done by the data collator
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    # Detailed classification report
    report = classification_report(labels, predictions, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'f1_macro': report['macro avg']['f1-score'],
        'f1_weighted': report['weighted avg']['f1-score'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall']
    }

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    logger.info("Starting BERT fine-tuning process")
    logger.info(f"Arguments: {vars(args)}")
    
    # Login to Hugging Face if token provided
    if args.hf_token:
        logger.info("Logging in to Hugging Face Hub")
        login(token=args.hf_token)
    
    # Load dataset
    dataset = load_and_prepare_dataset(args.dataset_name, args.feature_column, args.label_column)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=args.num_labels
    )
    
    # Tokenize datasets
    logger.info("Tokenizing datasets")
    
    # Get columns to remove (keep only label column and tokenized features)
    columns_to_remove = [col for col in dataset['train'].column_names if col != args.label_column]
    
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.feature_column, args.max_length),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    # Rename label column to 'labels' (required by Trainer)
    tokenized_datasets = tokenized_datasets.rename_column(args.label_column, 'labels')
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id if args.push_to_hub else None,
        report_to=None,  # Disable wandb/tensorboard reporting
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        seed=args.seed,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Print evaluation results
    logger.info("Final Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save evaluation results
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Generate detailed classification report on validation set
    logger.info("Generating detailed classification report...")
    predictions = trainer.predict(tokenized_datasets['validation'])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Save detailed report
    with open(output_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Final Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"Final F1 (Macro): {eval_results['eval_f1_macro']:.4f}")
    
    if args.push_to_hub and args.hub_model_id:
        logger.info(f"Model pushed to Hub: {args.hub_model_id}")
    
    # Save training configuration
    config = {
        'model_id': args.model_id,
        'dataset_name': args.dataset_name,
        'feature_column': args.feature_column,
        'label_column': args.label_column,
        'num_labels': args.num_labels,
        'training_args': training_args.to_dict(),
        'final_results': eval_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training configuration saved to training_config.json")
    logger.info("Training process completed!")

if __name__ == "__main__":
    main()
