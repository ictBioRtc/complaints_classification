# BERT Text Classification Workshop

This workshop guides you through training and using a BERT model for text classification using a complaint categories dataset. You'll learn how to fine-tune a pre-trained model and deploy it for both single and batch predictions.

## Prerequisites

- Python 3.8 or higher installed
- Basic understanding of Python
- Basic familiarity with command line/terminal
- A HuggingFace account (will create in Step 1)

## Step 1: HuggingFace Account Setup

1. Visit [HuggingFace](https://huggingface.co/) and click "Sign Up" if you don't have an account
2. After logging in, go to your Profile Settings
3. Navigate to "Access Tokens" (or visit https://huggingface.co/settings/tokens)
4. Click "New token"
5. Give it a name (e.g., "workshop-token")
6. **Important**: Select "Write" role to enable model uploading
7. Copy your token and keep it safe - you'll need it later!

## Step 2: Setting Up the Project

1. Open your terminal:
   ```bash
   # If using VS Code
   File > New Launcher > Terminal
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/ictBioRtc/complaints_classification.git
   ```

3. Navigate to the project directory:
   ```bash
   cd complaints_classification
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Step 3: Running the Application

1. Start the application:
   ```bash
   python app.py
   ```

2. The Gradio interface will launch automatically
3. You'll see a URL like `http://127.0.0.1:7860` in the terminal
4. Open this URL in your browser if it doesn't open automatically

## Step 4: Training the Model

1. In the Gradio interface, you'll see two tabs: "Train Model" and "Classify Complaints"
2. On the "Train Model" tab:
   - Dataset Name: `ictbiortc/complaint-categories-dataset`
   - Number of Epochs: Leave at default (3) for the workshop
   - Batch Size: Leave at default (8)
   - Learning Rate: Leave at default (2e-5)

3. If you want to save your model to HuggingFace:
   - Expand "Hugging Face Hub Settings"
   - Enter your HuggingFace token
   - Check "Push Model to Hub during training" or "Push trained model to Hub after training"
   - Enter your HuggingFace username
   - Enter a model name (e.g., "bert-complaint-classifier")

4. Click "Start Training"
   - Training will take several minutes
   - You'll see progress updates in the output box

## Step 5: Using the Model

1. Once training is complete, switch to the "Classify Complaints" tab
2. You have two options for classification:

   a. Single Complaint:
   - Enter text directly into the input box
   - Click "Classify"
   - View the predicted category

   b. Batch Processing:
   - Use the provided `test_complaints.csv` file
   - Upload it using the "Upload CSV" button
   - Click "Classify All"
   - View predictions for all complaints

3. Verify predictions against ground truth in `test_categories.csv`

## Using Your Own Dataset

To adapt this for your own dataset, ensure it follows the same format as `test_complaints.csv`:
- Must have a 'complaint' column
- Categories should match those in `test_categories.csv`
- Data should be in CSV format

## Categories

The model classifies text into three categories:
- Online-Safety
- BroadBand
- TV-Radio

## Troubleshooting

Common issues and solutions:
1. "Token not working": Ensure you selected "Write" permissions when creating the token
2. "Module not found": Run `pip install -r requirements.txt` again
3. "CUDA out of memory": Reduce batch size in training parameters
4. "Connection error": Check your internet connection

## Next Steps

After completing the workshop:
1. Try with your own dataset
2. Experiment with different training parameters
3. Test the model with different types of text input
4. Share your model on HuggingFace Hub

## Support

For questions during the workshop:
- Raise your hand to get the instructor's attention
- Check the error messages in the terminal
- Refer to this README for step-by-step guidance

---

Congratulations! You've now trained and deployed an AI model for text classification. This same approach can be adapted for various text classification tasks using your own datasets.
