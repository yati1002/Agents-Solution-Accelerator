# Databricks Healthcare Agent 🚑

## Overview
This repository contains a **Databricks solution accelerator** implementing a Healthcare-specific AI agent using the Databricks Lakehouse platform. It integrates structured and unstructured healthcare data, enabling tool-calling workflows and AI-powered insights.

## Features
- **Lakehouse integration**: Demonstrates how to load, clean, and join patient-centric datasets.
- **Healthcare agent**: NLP-driven conversational or retrieval agent tailored to clinical use cases.
- **Notebook-driven workflow**: Explains end-to-end use—from ingestion to application.
- **ML components**: Showcases embeddings, vector stores, LLM integration, and clinical prompt tuning.

## Blog Post
- `HOLDER`

### Prerequisites
- A Databricks Unity Catalog enabled workspace
- Serverless compute

### Installation
Clone the feature branch:

```bash
git clone -b feature/bcheng004 https://github.com/bcheng004/databricks-heathcare-agent.git
cd databricks-heathcare-agent
```

### Instructions
- In the `agents-solution-accelerator` folder navigate to the `data_setup`
- Make sure to replace `catalog` and `schema` name to your own catalog and schema prompted (*this applies to all steps*)
- Run `01_Setup Data` notebook using serverless compute
- Run `02_Create Vector Index` using serverless compute
- Navigate to the `agents-solution-accelerator` parent directory
- Run `single-agent-driver-multi-tool` using serverless compute
- Adjust `databricks_apps/app.yaml` env variables as necessary (SQL Warehouse, agent serving endpoint)
- [Deploy Databricks App](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/deploy#deploy-the-app)
