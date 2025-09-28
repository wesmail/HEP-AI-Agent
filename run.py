# Copyright (c) 2025 Waleed Esmail
# Licensed under the MIT License. See LICENSE file in the project root for details.

# hep_agent.py
# A minimal LangChain agent for HEP EDA + simple binary classification.
# Tools available to the agent inside the Python REPL: uproot, awkward, vector,
# pandas, pyarrow, seaborn/matplotlib, scikit-learn, torch (assumed preinstalled).
from dotenv import load_dotenv
from typing import Optional, List, Union
import json
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI  # swap to your provider if needed

load_dotenv()

# ---------- SYSTEM PROMPT (concise & generic) ----------
SYSTEM_PROMPT = (
    "You are a HEP EDA & ML helper. Inputs: two ROOT files (signal & background) "
    "with a 'Delphes' TTree. Tools: Python REPL with uproot, awkward, vector, "
    "pandas, pyarrow, seaborn/matplotlib, scikit-learn, torch. "
    "Goal: load data, engineer simple dijet/event features, make readable EDA plots, "
    "and train a small PyTorch MLP for BINARY classification.\n"
    "Rules:\n"
    "1) Confirm plan briefly, then RUN Python code via the REPL. Save artifacts: "
    "parquet dataset, eda_features.png, model.pt, training_metrics.json (and ROC PNG).\n"
    "2) Default engineered features (leading-two jets): pt, eta, phi, mass, btag; "
    "derived: m_jj, ΔR, |Δφ|, HT2=pt1+pt2; plus n_jets, met. Require ≥2 jets.\n"
    "3) EDA: stacked histograms (signal vs background) with correct units and large fonts.\n"
    "4) ML: split 70/15/15; standardize on train only; MLP depth/width from user; "
    "one-logit output + BCEWithLogitsLoss; Adam; early stopping; report accuracy & ROC-AUC.\n"
    "5) Be concise. Never install packages or run shell commands. If something is missing, say so.\n"

    "Reference example for reading + features (adapt paths/labels as needed):\n"
    "import numpy as np, pandas as pd, uproot, awkward as ak, vector\n"
    "def _lead2(x):\n"
    "    x = ak.pad_none(x, 2, clip=True)\n"
    "    x = ak.fill_none(x, 0)\n"
    "    return ak.to_numpy(x[:, :2]).astype(np.float32)\n"
    "def _first_or0(x):\n"
    "    return np.asarray(ak.firsts(ak.fill_none(ak.pad_none(x,1,clip=True),0)), np.float32)\n"
    "def load_make_df(path, tree='Delphes', label=0):\n"
    "    cols = ['Jet.PT','Jet.Eta','Jet.Phi','Jet.Mass','Jet.BTag','MissingET.MET']\n"
    "    arr = uproot.open(path)[tree].arrays(cols, library='ak')\n"
    "    mask = ak.num(arr['Jet.PT']) >= 2\n"
    "    arr = arr[mask]\n"
    "    pt  = _lead2(arr['Jet.PT']);   eta = _lead2(arr['Jet.Eta'])\n"
    "    phi = _lead2(arr['Jet.Phi']);  m   = _lead2(arr['Jet.Mass'])\n"
    "    b   = _lead2(ak.values_astype(arr['Jet.BTag'], np.float32))\n"
    "    nj  = np.asarray(ak.num(arr['Jet.PT']), np.int16)\n"
    "    met = _first_or0(arr['MissingET.MET'])\n"
    "    v1 = vector.array({{'pt':pt[:,0],'eta':eta[:,0],'phi':phi[:,0],'mass':m[:,0]}})\n"
    "    v2 = vector.array({{'pt':pt[:,1],'eta':eta[:,1],'phi':phi[:,1],'mass':m[:,1]}})\n"
    "    dR   = np.asarray(v1.deltaR(v2), np.float32)\n"
    "    dphi = np.abs(np.asarray(v1.deltaphi(v2), np.float32))\n"
    "    mjj  = np.asarray((v1+v2).mass, np.float32)\n"
    "    ht2  = (pt[:,0] + pt[:,1]).astype(np.float32)\n"
    "    return pd.DataFrame({{\n"
    "        'jet1_pt':pt[:,0], 'jet2_pt':pt[:,1],\n"
    "        'jet1_eta':eta[:,0], 'jet2_eta':eta[:,1],\n"
    "        'jet1_phi':phi[:,0], 'jet2_phi':phi[:,1],\n"
    "        'jet1_mass':m[:,0],  'jet2_mass':m[:,1],\n"
    "        'jet1_btag':b[:,0],  'jet2_btag':b[:,1],\n"
    "        'm_jj':mjj, 'dR_jj':dR, 'dphi_jj':dphi, 'HT2':ht2,\n"
    "        'n_jets':nj, 'met':met, 'y':np.full(len(mjj), label, np.int8),\n"
    "    }})\n"
)


# ---------- USER MESSAGE TEMPLATE ----------
# The agent expects a single JSON blob; if fields are missing, it uses defaults.
USER_TEMPLATE = """\
JSON config for the task:
{config_json}

If any paths or settings are missing, pick sensible defaults and proceed.
"""


def make_agent(model: str = "gpt-4o-mini", temperature: float = 0.0) -> AgentExecutor:
    """Create the LangChain agent with a Python REPL tool."""
    python_repl = PythonREPLTool()
    tools = [
        Tool(
            name="python",
            func=python_repl.run,
            description=(
                "Execute Python code to read ROOT files with uproot/awkward, "
                "engineer features, create seaborn EDA plots, and train a PyTorch MLP. "
                "Return printed outputs and file paths to saved artifacts."
            ),
        )
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", USER_TEMPLATE),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model=model, temperature=temperature)
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ---------- HELPER TO RUN THE AGENT ----------
def run_hep_agent(
    signal_path: str,
    background_path: str,
    tree_name: str = "Delphes",
    features: Union[str, List[str]] = "auto",  # "auto" -> engineer defaults
    hidden_layers: int = 2,
    hidden_units: Union[int, List[int]] = 64,  # int or list
    batch_size: int = 256,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    density_plots: bool = False,  # True -> density, False -> counts
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Build a config JSON and invoke the agent.
    Returns the agent's output dict (often includes printed text).
    """
    cfg = {
        "signal_path": signal_path,
        "background_path": background_path,
        "tree_name": tree_name,
        "features": features,
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "density_plots": density_plots,
    }

    agent = make_agent(model=model)
    result = agent.invoke(
        {"config_json": json.dumps(cfg, indent=2), "chat_history": []}
    )
    return result


# ---------- EXAMPLE USAGE ----------
if __name__ == "__main__":
    # Replace with your real file paths
    out = run_hep_agent(
        signal_path="Hbb.root",
        background_path="QCD.root",
        hidden_layers=2,
        hidden_units=[64, 64],
        epochs=20,
    )
    print("\n=== Agent Result ===")
    print(out.get("output", ""))
