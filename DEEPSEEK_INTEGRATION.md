# DeepSeek Integration

## Overview

This project uses DeepSeek (`deepseek-v4-pro`) as the LLM backend via the OpenAI-compatible API.

## Configuration

```python
OPENAI_API_KEY="your_api_key_here"
OPENAI_BASE_URL="https://api.deepseek.com/v1"
OPENAI_MODEL="deepseek-v4-pro"
```

## Features

- OpenAI-compatible chat completion API
- Temperature: 0 (deterministic outputs for analysis)
- Max tokens: 4000 (analysis), 2000 (general), 1000 (visualization)
- Built-in retry logic for rate limiting and message length errors

## Generalized Molecular Separation

This system supports separation analysis for **any two guest molecules**, not just CH4/CO2.

### Dual-Mode Query Routing

| Mode | Trigger Examples | Behavior |
|------|-----------------|----------|
| **Ranking** | "Which zeolite is best for X/Y separation?", "推荐分离X和Y最好的分子筛" | Full table scan → temperature pairing → Log10(D_max/D_min) ranking → Top-N recommendation |
| **Comparison** | "How different are X and Y in ZSM-5?", "X和Y在ZSM-5上扩散差多少?" | Look at specified zeolite/condition → direct diffusion coefficient comparison → ratio answer |

### Smart Column Mapping (3-Layer Framework)

1. **Layer 1 — Data Type**: Rule-based classification (numeric/categorical/text)
2. **Layer 2 — LLM Semantic Inference**: LLM infers business role from column name + samples
3. **Layer 3 — Role-to-Task Mapping**: Maps business roles to analysis task requirements

This means you can upload any CSV with different column names and the system adapts automatically.

## Supported Query Types

1. **Zeolite separation ranking**: "Which zeolite is best for separating methane and ethane?"
2. **Direct comparison**: "How different are CO2 and N2 diffusion in ZSM-5?"
3. **General data analysis**: "Show me all data above 300K"
4. **Visualization**: "Plot temperature vs diffusion coefficient"
5. **Chinese language**: "推荐分离二氧化碳和氮气最好的分子筛"
