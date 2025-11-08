# Chatbot Training Data Guide

## How Training Data is Generated

The chatbot training system automatically generates training data from your product dataset. Here's how it works:

### Automatic Generation (Recommended)

**On First Run:**
1. When you start the app, the chatbot checks if training data exists
2. If training data is missing, it automatically generates:
   - **Knowledge Base** (`training_data/knowledge_base.json`): Extracts product information from CSV
   - **Training Prompts** (`training_data/training_prompts.json`): Creates training examples

**What Gets Generated:**
- Product information organized by category, brand, and type
- 315+ brands cataloged
- 28,000+ products indexed
- 142+ training prompts covering:
  - Product searches
  - Greetings and basic communication
  - Help requests
  - Brand inquiries

### Manual Generation (Optional)

You can also manually generate training data by running:

```bash
python3 chatbot_trainer.py
```

This is useful if:
- You've updated the CSV dataset
- You want to regenerate training data
- You want to see the generation process

## Running on a New Device

### What Happens Automatically

When you run the app on a new device:

1. **First Launch:**
   - App detects missing training data
   - Automatically reads the CSV file (`home appliance skus lowes.csv`)
   - Generates knowledge base (takes ~10-30 seconds)
   - Generates training prompts
   - Saves everything to `training_data/` folder
   - Chatbot is ready to use!

2. **Subsequent Launches:**
   - App loads existing training data (instant)
   - No regeneration needed

### Requirements for New Device

To run on a new device, you need:

1. **CSV Dataset:**
   - `home appliance skus lowes.csv` must be in the project root
   - This is the source data for training

2. **Python Dependencies:**
   - All packages from `requirements.txt` installed
   - pandas (for CSV processing)
   - json (built-in)

3. **File Structure:**
   ```
   project-root/
   ├── home appliance skus lowes.csv  ← Required
   ├── chatbot_trainer.py
   ├── chatbot.py
   └── training_data/                 ← Auto-created
       ├── knowledge_base.json        ← Auto-generated
       └── training_prompts.json      ← Auto-generated
   ```

### Training Data Files

**`training_data/knowledge_base.json`**
- Contains product information extracted from CSV
- Organized by category, brand, and product type
- Includes price ranges and product counts
- Size: ~10-50 MB (depending on dataset size)

**`training_data/training_prompts.json`**
- Contains training examples for the chatbot
- Includes greetings, product queries, and help requests
- Used to improve chatbot responses
- Size: ~100-500 KB

### Performance

**First Run (New Device):**
- Knowledge base generation: 10-30 seconds
- Training prompts generation: <1 second
- Total: ~10-30 seconds

**Subsequent Runs:**
- Loading existing data: <1 second
- No delay

### Troubleshooting

**If training data doesn't generate:**

1. **Check CSV file exists:**
   ```bash
   ls -la "home appliance skus lowes.csv"
   ```

2. **Check permissions:**
   - Ensure you have read access to CSV
   - Ensure you have write access to create `training_data/` folder

3. **Check logs:**
   - Look for error messages in console
   - Check if CSV file is valid

4. **Manual generation:**
   ```bash
   python3 chatbot_trainer.py
   ```

**If training data is corrupted:**

- Delete `training_data/` folder
- Restart app (will auto-regenerate)

## Best Practices

1. **Version Control:**
   - Training data is in `.gitignore` (not committed)
   - Each device generates its own training data
   - This is intentional - keeps repo clean

2. **Updating Dataset:**
   - If you update the CSV file, delete `training_data/` folder
   - Restart app to regenerate with new data

3. **Sharing Training Data (Optional):**
   - You can copy `training_data/` folder to another device
   - Saves regeneration time
   - Must match the same CSV file structure

## Summary

✅ **Automatic:** Training data generates automatically on first run  
✅ **Fast:** Subsequent runs load instantly  
✅ **Portable:** Works on any device with CSV file  
✅ **Self-Healing:** Regenerates if missing or corrupted  

No manual intervention needed! Just run the app and it handles everything.

