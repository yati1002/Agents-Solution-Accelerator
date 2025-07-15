import logging
import os
import streamlit as st
from model_serving_utils import query_endpoint
import json
import re
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
assert os.getenv("SERVING_ENDPOINT"), "SERVING_ENDPOINT must be set in app.yaml."
assert os.getenv(
    "DATABRICKS_WAREHOUSE_ID"
), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

st.set_page_config(layout="wide")


def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
        # Can use st.context.ip_address to get the user's IP address if not on localhost
        user_ip_address="66.249.65.103",
    )


user_info = get_user_info()


def sqlQuery(query: str) -> pd.DataFrame:
    cfg = Config()  # Pull environment variables for auth
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


@st.cache_data(ttl=30)  # only re-query if it's been 30 seconds
def getMemberIds():
    # This example query depends on the nyctaxi data set in Unity Catalog, see https://docs.databricks.com/en/discover/databricks-datasets.html for details
    return sqlQuery(
        "select distinct member_id from hls_yatish.agent_solution_accelerator.member_enrolment"
    )


member_id_data = getMemberIds()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("🏥 Chat with your Healthcare Data")
st.markdown(
    "ℹ️ This is a simple example. See "
    "[Databricks docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app) "
    "for a more comprehensive example with streaming output and more."
)
member_id_option = st.selectbox(
    "Member Id", sorted(member_id_data["member_id"].tolist()), index=0
)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
# {st.context.ip_address} is None bc docker image is running on localhost
ip_address_partial_prompt = f"my IP address is {user_info['user_ip_address']}"
member_id_partial_prompt = f"my member id is {member_id_option}."
if prompt := st.chat_input(f"What medical issue do you need information on today?"):
    prompt = f"{prompt} {ip_address_partial_prompt}"
    # Add user message to chat history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"{member_id_partial_prompt} {prompt} {ip_address_partial_prompt}",
        }
    )
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(
            prompt.replace(ip_address_partial_prompt, "").replace(
                member_id_partial_prompt, ""
            )
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Query the Databricks serving endpoint
        endpoint_response = query_endpoint(
            endpoint_name=os.getenv("SERVING_ENDPOINT"),
            messages=st.session_state.messages,
            max_tokens=2048,
            temperature=0.1,
        )
        assistant_response = endpoint_response[0]["content"]
        formatted_response = json.loads(
            endpoint_response[-1].get("tool_calls")[0].get("function").get("arguments")
        )
        lat_long_patttern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        latitude = formatted_response.get("first_doctor_lattitude")
        latitiude_sign = 1
        if "S" in latitude:
            latitiude_sign = -1
        latitude = float(re.findall(lat_long_patttern, latitude)[0]) * latitiude_sign
        longitude = formatted_response.get("first_doctor_longitude")
        longitude_sign = 1
        if "W" in longitude:
            longitude_sign = -1
        longitude = float(re.findall(lat_long_patttern, longitude)[0]) * longitude_sign
        # st.text(f"latitude: {latitude}")
        # st.text(f"longitude: {longitude}")
        st.text(assistant_response)

        df = pd.DataFrame([[latitude, longitude]], columns=["lat", "lon"])
        st.map(df)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )