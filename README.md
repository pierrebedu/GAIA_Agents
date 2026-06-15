# Presentation du repo

This is an experiment on the popular GAIA benchmark (static questions, as opposed to GAIA2).  https://arxiv.org/pdf/2311.12983 \
Create and improve an agent to answer questions on topics from the web/wikipedia/given files. \


# Venv  sur factorius

source .venv/bin/activate

# Pour reutiliser ailleurs
python3 -m venv .venv \
source .venv/bin/activate \
pip install -r requirements.txt \




explore_gaia.ipynb is an EDA on the dataset. \

regexs.py are utils fonctions used in evaluate_agent.py & explore_gaia.ipynb \

evaluate_agents.py gets 20 questions from the GAIA test set and submits all the answers to a HF server. Answers are scored. \

# MAIN CODE :
main agent is build in agent.py and follows the systemp_prompt.txt inbstructions. \

created with langgraph with a "create_react_agent(llm, tools, system_prompt)" instruction.

# Features :
- has a tool to get youtube transcripts.
- has a retries (2) system in case tool calling fails
- tools : search the web/search wikipedia/ search arxiv / execute python code / transcripts youtube video

# traces
Traced with langfuse cloud (remark : possible to self-host it).


TO DO :
explain what runs under the hood in the react_agent instance.