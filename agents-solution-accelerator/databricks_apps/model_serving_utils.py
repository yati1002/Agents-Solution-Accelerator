from mlflow.deployments import get_deploy_client


def _query_endpoint(
    endpoint_name: str, messages: list[dict[str, str]], max_tokens, temperature
) -> list[dict[str, str]]:
    """Calls a model serving endpoint."""
    res = get_deploy_client("databricks").predict(
        endpoint=endpoint_name,
        inputs={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    if "messages" in res:
        return res["messages"]
    elif "choices" in res:
        return [res["choices"][0]["message"]]
    raise Exception(
        "This app can only run against:"
        "1) Databricks foundation model or external model endpoints with the chat task type (described in https://docs.databricks.com/aws/en/machine-learning/model-serving/score-foundation-models#chat-completion-model-query)"
        "2) Databricks agent serving endpoints that implement the conversational agent schema documented "
        "in https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent"
    )


def query_endpoint(endpoint_name, messages, max_tokens, temperature):
    """
    Query a chat-completions or agent serving endpoint
    If querying an agent serving endpoint that returns multiple messages, this method
    returns the last message
    ."""
    response = _query_endpoint(endpoint_name, messages, max_tokens, temperature)
    return (
        response[-1],
        [
            message
            for message in response
            if message.get("tool_calls") is not None
            and message.get("tool_calls")[0].get("function") is not None
            and message.get("tool_calls")[0].get("function").get("name")
            == "ResponseFormatter"
        ][0],
    )