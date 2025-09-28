# HEP AI Agent

This is a proof of concept minimal **AI agent** for High Energy Physics (HEP) and **simple ML tasks**.  
It uses a Python REPL tool with libraries like **uproot, awkward, vector, pandas, seaborn, scikit-learn, torch** to:

- Load signal & background ROOT files (`Delphes` TTree).  
- Engineer simple event features.  
- Produce clear EDA plots (stacked histograms).  
- Train a small PyTorch MLP (with user-specified depth/width).  
- Save artifacts: `dataset.parquet`, `eda_features.png`, `model.pt`, `training_metrics.json`, and ROC curve.  

This agent relies on **GPT-4**, which requires an OpenAI API key.  
You can set up billing with a minimal monthly limit (e.g., $5), which is typically more than sufficient for running this project.  

## Getting an OpenAI API Key (minimum cost setup)

Follow these steps to get your own OpenAI API key and spend the least possible:

### 1. Create an OpenAI Account
1. Go to [https://platform.openai.com/](https://platform.openai.com/).
2. Sign up with your email, Google, or Microsoft account.
3. Verify your email account.

### 2. Access the API Dashboard
To use this agent, you need an **OpenAI API key**. Follow these steps:

2.1. **Log in to OpenAI**  
   Go to [https://platform.openai.com/](https://platform.openai.com/) and log in with your account.  

2.2. **Go to the API Keys page**  
   Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).  
   Here you will see the option to manage your API keys.  

2.3. **Create a new key**  
   - Click on **“Create new secret key”**.  
   - A long string starting with `sk-` will be shown (e.g., `sk-xxxxxxxxxxxxxxxx`).  
   - **Important:** Copy this key immediately — you won’t be able to see it again once you close the popup.  

2.4. **Store your key securely**  
   - In your project folder, create a file named `.env`.  
   - Open it with a text editor and add the following line (replace with your actual key):  

     ```bash
     OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
     ```
   - Save and close the file.  

2.5. **Why the `.env` file?**  
   The `.env` file keeps your API key separate from your code, so you don’t accidentally share it when publishing your project.  
   The agent automatically loads the key from this file when you run it.


### 3. Add Billing with Minimal Spend
1. Go to [Billing Settings](https://platform.openai.com/account/billing/overview).
2. Add a **payment method** (credit card or PayPal).
3. Set a **low monthly usage limit**:
   - In **Billing → Usage limits**, set a **hard limit** (e.g. `$5/month`) and a **soft limit** (e.g. `$2/month`).
   - This ensures you will not exceed your budget.  

You can use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package manager) to create an isolated environment for the agent.

## Install `uv` (if not already installed)  
```bash
# Recommended: install uv in your home directory
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After installation, make sure uv is in your PATH. You can check with:  
```bash
uv --version
```
## Create a new virtual environment
```bash
# Create a new environment
uv init hep-agent

# Activate the environment
source hep-agent/bin/activate   # Linux/macOS
hep-agent\Scripts\activate      # Windows (PowerShell)
```
## Install project dependencies  
```bash
uv add uproot awkward vector pandas pyarrow seaborn matplotlib scikit-learn torch python-dotenv langchain langchain-openai
```

This agent requires two ROOT files as input: the output of Delphes, which contains a `TTree` named `Delphes`.
For demonstration, the agent is configured to use:

- `Hbb.root` → the signal sample (H → bb)
- `QCD.root` → the background sample

To run the agent, simply execute:
```bash
ipython run.py
```
For convenience, the repository includes [dummy ROOT files](https://drive.google.com/drive/folders/1wRfgNBrvD4TZjtiptpfgCtKhGcNO4AS3?usp=drive_link) (`Hbb.root` and `QCD.root`), each containing 1,000 events, so you can test the workflow out of the box.
