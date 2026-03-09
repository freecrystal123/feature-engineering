# Databricks notebook source
LLM_ATTEMPTS = 3

# COMMAND ----------

from databricks_genai_inference import ChatCompletion

def get_llm_response(analysis:str, prompt:str, user_query:str):
    try:
        analysis_name = ANALYSIS_NAMES[analysis]
    except:
        analysis_name = "test"

    ctx = prompt.replace("<<SUBJECT>>", analysis_name)



    response = ChatCompletion.create(
        # model="databricks-llama-4-maverick",
        # model="databricks-gpt-5-2",
        model="databricks-claude-opus-4-6",
        messages=[
            {
                "role": "system",
                "content": ctx
            },
            {
                "role": "user",
                "content": user_query
            },
            {
                "role": "system",
                "content": ctx
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        max_tokens=3000
    )
    return response.message

# COMMAND ----------

# MAGIC %skip
# MAGIC ANALYSIS_NAMES = {'analysis_1':'online casino'}

# COMMAND ----------

# %skip
# get_llm_response('analysis_1',"E",'whats 2+2?')

# COMMAND ----------

import re
from collections import defaultdict

def extract_variables_from_block(text, variable_names, flag_several=True):
    """
    Extract variables inside <<<RESULT>>> ... <<<END>>> blocks.

    Supports:
      - Indexed variables: VAR_1=value
      - Non-indexed variables: VAR=value

    Args:
        text (str): Full input text.
        variable_names (list): Variable names or prefixes (e.g. ["PCA_NAME_", "RATIONALE_"]
                               or ["CLUSTER_NAME", "CLUSTER_DESCRIPTION"]).
        flag_several (bool): If True, returns all indices.
                             If False, returns only the first (still as list).

    Returns:
        list[dict]
    """

    # 1. Extract RESULT block
    block_match = re.search(r'<<<RESULT>>>(.*?)<<<END>>>', text, re.S)
    if not block_match:
        return []

    block = block_match.group(1)

    grouped = defaultdict(dict)
    has_index = False

    for var in variable_names:
        # Case 1: indexed variable (ends with _)
        if var.endswith("_"):
            pattern = re.compile(rf'{re.escape(var)}(\d+)\s*=\s*(.+)')
            matches = pattern.findall(block)
            if matches:
                has_index = True
            for idx, value in matches:
                key = var.rstrip("_")
                grouped[idx][key] = value.strip()
                grouped[idx]["index"] = idx

        # Case 2: non-indexed variable
        else:
            pattern = re.compile(rf'{re.escape(var)}\s*=\s*(.+)')
            match = pattern.search(block)
            if match:
                grouped["1"][var] = match.group(1).strip()
                grouped["1"]["index"] = "1"

    results = [grouped[idx] for idx in sorted(grouped, key=lambda x: int(x))]

    if not flag_several and results:
        return [results[0]]

    return results

# COMMAND ----------

def rename_pca(analysis: str, old_name: str, new_name: str, new_description: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    table_name = f"{T_GOLD_CUSTOMER_PCA_COMP_PREFIX}_{analysis_name}"
    query = f"""
    UPDATE {table_name}
    SET pca_component_name = '{new_name}', pca_component_description = '{new_description}'
    WHERE upper(pca_component_name) = upper('{old_name}')
    """
    spark.sql(query)

# COMMAND ----------

def rename_cluster(analysis: str, old_name: str, new_name: str, new_description: str):
    analysis_name = ANALYSIS_NAMES[analysis]
    table_name = f"{T_GOLD_CUSTOMER_CLUSTERS_PREFIX}_{analysis_name}"
    query = f"""
    UPDATE {table_name}
    SET cluster_name_refined = '{new_name}', cluster_description_refined = '{new_description}'
    WHERE upper(cluster_name_refined) = upper('{old_name}')
    """
    spark.sql(query)