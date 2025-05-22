# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow langchain langgraph==0.3.4 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """You are a friendly and professional customer service agent for a car service center.
# MAGIC  
# MAGIC Always greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.
# MAGIC  
# MAGIC Once you have the mobile number:
# MAGIC  
# MAGIC 1. Check if the customer exists
# MAGIC    - If the customer exists:
# MAGIC      - Retrieve their details and associated vehicles.
# MAGIC      - Greet them by name.
# MAGIC      - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).
# MAGIC    - If the customer does NOT exist:
# MAGIC      - Let them know you’ll help get them registered.
# MAGIC      - Ask for their full name, phone number, and email.
# MAGIC      - Insert their details into the `customer` table.
# MAGIC      - Then ask about their car (model, plate number, year) 
# MAGIC  
# MAGIC After registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:
# MAGIC - Book an appointment
# MAGIC - Modify or cancel an existing booking
# MAGIC - Register a new vehicle
# MAGIC - Check service history
# MAGIC Always be courteous, ask one question at a time, and confirm key actions."""
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC uc_tool_names = ["datalink.lineagedemo.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC
# MAGIC # # (Optional) Use Databricks vector search indexes as tools
# MAGIC # # See https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools
# MAGIC # # for details
# MAGIC #
# MAGIC # # TODO: Add vector search indexes as tools or delete this block
# MAGIC # vector_search_tools = [
# MAGIC #         VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # ]
# MAGIC # tools.extend(vector_search_tools)
# MAGIC
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Hello!"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What is 5+5 in python"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog tool queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

# TODO: Manually include underlying resources if needed. See the TODO in the markdown above for more information.
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "Hi,i'm varun wanted to book an service"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        extra_pip_requirements=[
            "databricks-connect"
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC To evaluate your tool calls, try adding [custom metrics](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls).

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.insert_vehicle_details\n\n<uc_function_call>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"name\":\"datalink.lineagedemo.insert_vehicle_details\",\"arguments\":\"{\\\"cutomer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"model\\\":\\\"Baleno V3\\\",\\\"plate_number\\\":\\\"KL14\\\",\\\"year\\\":\\\"2023\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Vehicle info created. ID: 9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.get_available_service\n\n<uc_function_call>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"name\":\"datalink.lineagedemo.get_available_service\",\"arguments\":\"{}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"service_type_id\\\",\\\"service_name\\\",\\\"description\\\"],\\\"rows\\\":[[\\\"S001\\\",\\\"Periodic Maintenance\\\",\\\"Routine maintenance at mileage intervals\\\"],[\\\"S002\\\",\\\"Engine Diagnostics\\\",\\\"Check engine and performance issues\\\"],[\\\"S003\\\",\\\"AC Repair\\\",\\\"Air conditioning service and repair\\\"]]}\"}</uc_function_result>\n\nThank you, Varun! I've successfully registered your 2023 Baleno V3 with plate number KL14 in our system.\n\nNow, let's proceed with booking your service appointment. We offer the following service types:\n\n1. Periodic Maintenance - Routine maintenance at mileage intervals\n2. Engine Diagnostics - Check engine and performance issues\n3. AC Repair - Air conditioning service and repair\n\nWhich type of service would you like to book for your Baleno? Also, do you have a preferred service center location?"
                },
                {
                    "role": "user",
                    "content": "AC Repair"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.insert_vehicle_details\n\n<uc_function_call>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"name\":\"datalink.lineagedemo.insert_vehicle_details\",\"arguments\":\"{\\\"cutomer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"model\\\":\\\"Baleno V3\\\",\\\"plate_number\\\":\\\"KL14\\\",\\\"year\\\":\\\"2023\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Vehicle info created. ID: 9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.get_available_service\n\n<uc_function_call>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"name\":\"datalink.lineagedemo.get_available_service\",\"arguments\":\"{}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"service_type_id\\\",\\\"service_name\\\",\\\"description\\\"],\\\"rows\\\":[[\\\"S001\\\",\\\"Periodic Maintenance\\\",\\\"Routine maintenance at mileage intervals\\\"],[\\\"S002\\\",\\\"Engine Diagnostics\\\",\\\"Check engine and performance issues\\\"],[\\\"S003\\\",\\\"AC Repair\\\",\\\"Air conditioning service and repair\\\"]]}\"}</uc_function_result>\n\nThank you, Varun! I've successfully registered your 2023 Baleno V3 with plate number KL14 in our system.\n\nNow, let's proceed with booking your service appointment. We offer the following service types:\n\n1. Periodic Maintenance - Routine maintenance at mileage intervals\n2. Engine Diagnostics - Check engine and performance issues\n3. AC Repair - Air conditioning service and repair\n\nWhich type of service would you like to book for your Baleno? Also, do you have a preferred service center location?"
                },
                {
                    "role": "user",
                    "content": "AC Repair"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for selecting AC Repair for your 2023 Baleno V3. \n\nTo proceed with booking your appointment, I'll need to know which service center location you prefer. This will help me check the available time slots at that specific center.\n\nCould you please let me know which service center you'd like to visit? We have multiple locations available."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.insert_vehicle_details\n\n<uc_function_call>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"name\":\"datalink.lineagedemo.insert_vehicle_details\",\"arguments\":\"{\\\"cutomer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"model\\\":\\\"Baleno V3\\\",\\\"plate_number\\\":\\\"KL14\\\",\\\"year\\\":\\\"2023\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Vehicle info created. ID: 9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.get_available_service\n\n<uc_function_call>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"name\":\"datalink.lineagedemo.get_available_service\",\"arguments\":\"{}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"service_type_id\\\",\\\"service_name\\\",\\\"description\\\"],\\\"rows\\\":[[\\\"S001\\\",\\\"Periodic Maintenance\\\",\\\"Routine maintenance at mileage intervals\\\"],[\\\"S002\\\",\\\"Engine Diagnostics\\\",\\\"Check engine and performance issues\\\"],[\\\"S003\\\",\\\"AC Repair\\\",\\\"Air conditioning service and repair\\\"]]}\"}</uc_function_result>\n\nThank you, Varun! I've successfully registered your 2023 Baleno V3 with plate number KL14 in our system.\n\nNow, let's proceed with booking your service appointment. We offer the following service types:\n\n1. Periodic Maintenance - Routine maintenance at mileage intervals\n2. Engine Diagnostics - Check engine and performance issues\n3. AC Repair - Air conditioning service and repair\n\nWhich type of service would you like to book for your Baleno? Also, do you have a preferred service center location?"
                },
                {
                    "role": "user",
                    "content": "AC Repair"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for selecting AC Repair for your 2023 Baleno V3. \n\nTo proceed with booking your appointment, I'll need to know which service center location you prefer. This will help me check the available time slots at that specific center.\n\nCould you please let me know which service center you'd like to visit? We have multiple locations available."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.get_available_slots\n\n<uc_function_call>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"name\":\"datalink.lineagedemo.get_available_slots\",\"arguments\":\"{\\\"center_name\\\":\\\"Jeddah\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nI apologize, but I don't see any service center named \"Jeddah\" in our system, or there might not be any available slots at that location currently.\n\nCould you please choose from one of our other service center locations? Unfortunately, I don't have the complete list of our centers in the system. Common locations might include downtown centers, mall locations, or centers in specific neighborhoods.\n\nAlternatively, if you know another name for the Jeddah center or have a different preferred location, please let me know."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.insert_vehicle_details\n\n<uc_function_call>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"name\":\"datalink.lineagedemo.insert_vehicle_details\",\"arguments\":\"{\\\"cutomer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"model\\\":\\\"Baleno V3\\\",\\\"plate_number\\\":\\\"KL14\\\",\\\"year\\\":\\\"2023\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Vehicle info created. ID: 9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.get_available_service\n\n<uc_function_call>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"name\":\"datalink.lineagedemo.get_available_service\",\"arguments\":\"{}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"service_type_id\\\",\\\"service_name\\\",\\\"description\\\"],\\\"rows\\\":[[\\\"S001\\\",\\\"Periodic Maintenance\\\",\\\"Routine maintenance at mileage intervals\\\"],[\\\"S002\\\",\\\"Engine Diagnostics\\\",\\\"Check engine and performance issues\\\"],[\\\"S003\\\",\\\"AC Repair\\\",\\\"Air conditioning service and repair\\\"]]}\"}</uc_function_result>\n\nThank you, Varun! I've successfully registered your 2023 Baleno V3 with plate number KL14 in our system.\n\nNow, let's proceed with booking your service appointment. We offer the following service types:\n\n1. Periodic Maintenance - Routine maintenance at mileage intervals\n2. Engine Diagnostics - Check engine and performance issues\n3. AC Repair - Air conditioning service and repair\n\nWhich type of service would you like to book for your Baleno? Also, do you have a preferred service center location?"
                },
                {
                    "role": "user",
                    "content": "AC Repair"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for selecting AC Repair for your 2023 Baleno V3. \n\nTo proceed with booking your appointment, I'll need to know which service center location you prefer. This will help me check the available time slots at that specific center.\n\nCould you please let me know which service center you'd like to visit? We have multiple locations available."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.get_available_slots\n\n<uc_function_call>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"name\":\"datalink.lineagedemo.get_available_slots\",\"arguments\":\"{\\\"center_name\\\":\\\"Jeddah\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nI apologize, but I don't see any service center named \"Jeddah\" in our system, or there might not be any available slots at that location currently.\n\nCould you please choose from one of our other service center locations? Unfortunately, I don't have the complete list of our centers in the system. Common locations might include downtown centers, mall locations, or centers in specific neighborhoods.\n\nAlternatively, if you know another name for the Jeddah center or have a different preferred location, please let me know."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I apologize for the confusion. It seems we're having trouble finding a service center named \"Jeddah\" in our system, or there may not be any available slots at that location currently.\n\nLet me try to help you in a different way. Instead of searching by the center name, could you tell me which general area or city you're located in? This might help us identify the nearest service center to you.\n\nAlternatively, would you like me to check if we have any other service centers with available slots? This would give us a starting point to book your AC Repair service."
                },
                {
                    "role": "user",
                    "content": "recheck once"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.insert_vehicle_details\n\n<uc_function_call>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"name\":\"datalink.lineagedemo.insert_vehicle_details\",\"arguments\":\"{\\\"cutomer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"model\\\":\\\"Baleno V3\\\",\\\"plate_number\\\":\\\"KL14\\\",\\\"year\\\":\\\"2023\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Vehicle info created. ID: 9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.get_available_service\n\n<uc_function_call>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"name\":\"datalink.lineagedemo.get_available_service\",\"arguments\":\"{}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"service_type_id\\\",\\\"service_name\\\",\\\"description\\\"],\\\"rows\\\":[[\\\"S001\\\",\\\"Periodic Maintenance\\\",\\\"Routine maintenance at mileage intervals\\\"],[\\\"S002\\\",\\\"Engine Diagnostics\\\",\\\"Check engine and performance issues\\\"],[\\\"S003\\\",\\\"AC Repair\\\",\\\"Air conditioning service and repair\\\"]]}\"}</uc_function_result>\n\nThank you, Varun! I've successfully registered your 2023 Baleno V3 with plate number KL14 in our system.\n\nNow, let's proceed with booking your service appointment. We offer the following service types:\n\n1. Periodic Maintenance - Routine maintenance at mileage intervals\n2. Engine Diagnostics - Check engine and performance issues\n3. AC Repair - Air conditioning service and repair\n\nWhich type of service would you like to book for your Baleno? Also, do you have a preferred service center location?"
                },
                {
                    "role": "user",
                    "content": "AC Repair"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for selecting AC Repair for your 2023 Baleno V3. \n\nTo proceed with booking your appointment, I'll need to know which service center location you prefer. This will help me check the available time slots at that specific center.\n\nCould you please let me know which service center you'd like to visit? We have multiple locations available."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.get_available_slots\n\n<uc_function_call>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"name\":\"datalink.lineagedemo.get_available_slots\",\"arguments\":\"{\\\"center_name\\\":\\\"Jeddah\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nI apologize, but I don't see any service center named \"Jeddah\" in our system, or there might not be any available slots at that location currently.\n\nCould you please choose from one of our other service center locations? Unfortunately, I don't have the complete list of our centers in the system. Common locations might include downtown centers, mall locations, or centers in specific neighborhoods.\n\nAlternatively, if you know another name for the Jeddah center or have a different preferred location, please let me know."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I apologize for the confusion. It seems we're having trouble finding a service center named \"Jeddah\" in our system, or there may not be any available slots at that location currently.\n\nLet me try to help you in a different way. Instead of searching by the center name, could you tell me which general area or city you're located in? This might help us identify the nearest service center to you.\n\nAlternatively, would you like me to check if we have any other service centers with available slots? This would give us a starting point to book your AC Repair service."
                },
                {
                    "role": "user",
                    "content": "recheck once"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.get_available_slots\n\n<uc_function_call>{\"id\":\"A0l4sIhNQTztd61DOxkR\",\"name\":\"datalink.lineagedemo.get_available_slots\",\"arguments\":\"{\\\"center_name\\\":\\\"Jeddah\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"A0l4sIhNQTztd61DOxkR\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"slot_id\\\",\\\"center_id\\\",\\\"location\\\",\\\"slot_datetime\\\",\\\"is_available\\\"],\\\"rows\\\":[[\\\"SL001\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T09:00:00.000+0000\\\",true],[\\\"SL002\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T11:00:00.000+0000\\\",true],[\\\"SL003\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T13:00:00.000+0000\\\",true],[\\\"SL004\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T15:00:00.000+0000\\\",true],[\\\"SL005\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T17:00:00.000+0000\\\",true],[\\\"SL016\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T09:00:00.000+0000\\\",true],[\\\"SL017\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T11:00:00.000+0000\\\",true],[\\\"SL018\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T13:00:00.000+0000\\\",true],[\\\"SL019\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T15:00:00.000+0000\\\",true],[\\\"SL020\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T17:00:00.000+0000\\\",true],[\\\"SL031\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T09:00:00.000+0000\\\",true],[\\\"SL032\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T11:00:00.000+0000\\\",true],[\\\"SL033\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T13:00:00.000+0000\\\",true],[\\\"SL034\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T15:00:00.000+0000\\\",true],[\\\"SL035\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T17:00:00.000+0000\\\",true],[\\\"SL046\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T09:00:00.000+0000\\\",true],[\\\"SL047\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T11:00:00.000+0000\\\",true],[\\\"SL048\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T13:00:00.000+0000\\\",true],[\\\"SL049\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T15:00:00.000+0000\\\",true],[\\\"SL050\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T17:00:00.000+0000\\\",true],[\\\"SL061\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T09:00:00.000+0000\\\",true],[\\\"SL062\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T11:00:00.000+0000\\\",true],[\\\"SL063\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T13:00:00.000+0000\\\",true],[\\\"SL064\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T15:00:00.000+0000\\\",true],[\\\"SL065\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T17:00:00.000+0000\\\",true],[\\\"SL076\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T09:00:00.000+0000\\\",true],[\\\"SL077\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T11:00:00.000+0000\\\",true],[\\\"SL078\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T13:00:00.000+0000\\\",true],[\\\"SL079\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T15:00:00.000+0000\\\",true],[\\\"SL080\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T17:00:00.000+0000\\\",true],[\\\"SL091\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T09:00:00.000+0000\\\",true],[\\\"SL092\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T11:00:00.000+0000\\\",true],[\\\"SL093\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T13:00:00.000+0000\\\",true],[\\\"SL094\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T15:00:00.000+0000\\\",true],[\\\"SL095\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T17:00:00.000+0000\\\",true],[\\\"SL106\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T09:00:00.000+0000\\\",true],[\\\"SL107\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T11:00:00.000+0000\\\",true],[\\\"SL108\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T13:00:00.000+0000\\\",true],[\\\"SL109\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T15:00:00.000+0000\\\",true],[\\\"SL110\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T17:00:00.000+0000\\\",true],[\\\"SL121\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T09:00:00.000+0000\\\",true],[\\\"SL122\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T11:00:00.000+0000\\\",true],[\\\"SL123\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T13:00:00.000+0000\\\",true],[\\\"SL124\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T15:00:00.000+0000\\\",true],[\\\"SL125\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T17:00:00.000+0000\\\",true],[\\\"SL136\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T09:00:00.000+0000\\\",true],[\\\"SL137\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T11:00:00.000+0000\\\",true],[\\\"SL138\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T13:00:00.000+0000\\\",true],[\\\"SL139\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T15:00:00.000+0000\\\",true],[\\\"SL140\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T17:00:00.000+0000\\\",true],[\\\"SL151\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T09:00:00.000+0000\\\",true],[\\\"SL152\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T11:00:00.000+0000\\\",true],[\\\"SL153\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T13:00:00.000+0000\\\",true],[\\\"SL154\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T15:00:00.000+0000\\\",true],[\\\"SL155\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T17:00:00.000+0000\\\",true]]}\"}</uc_function_result>\n\nThank you for your patience! I've successfully found available slots at our Jeddah - City Road service center. Here are some of the available appointment times:\n\nFor May 20, 2025:\n- 9:00 AM\n- 11:00 AM\n- 1:00 PM\n- 3:00 PM\n- 5:00 PM\n\nFor May 21, 2025:\n- 9:00 AM\n- 11:00 AM\n- 1:00 PM\n- 3:00 PM\n- 5:00 PM\n\nWe also have similar time slots available on subsequent days through May 30, 2025.\n\nWhich date and time would you prefer for your AC Repair service appointment?"
                },
                {
                    "role": "user",
                    "content": "Book for May 20 9:00 AM"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a friendly and professional customer service agent for a car service center.\n \nAlways greet the customer politely when the conversation begins. Then ask for their mobile number if it hasn't been provided yet.\n \nOnce you have the mobile number:\n \n1. Check if the customer exists\n   - If the customer exists:\n     - Retrieve their details and associated vehicles.\n     - Greet them by name.\n     - Ask how you can help today (e.g., book a service, modify booking, cancel, register a new car, check service history).\n   - If the customer does NOT exist:\n     - Let them know you’ll help get them registered.\n     - Ask for their full name, phone number, and email.\n     - Insert their details into the `customer` table.\n     - Then ask about their car (model, plate number, year) \n \nAfter registration or customer verification, continue the conversation to understand the purpose of the call and help them accordingly:\n- Book an appointment\n- Modify or cancel an existing booking\n- Register a new vehicle\n- Check service history\nAlways be courteous, ask one question at a time, and confirm key actions."
                },
                {
                    "role": "user",
                    "content": "Hi,i'm varun wanted to book an service"
                },
                {
                    "role": "assistant",
                    "content": "Hello Varun! Thank you for contacting our car service center. I'd be happy to help you book a service appointment.\n\nTo get started, could you please provide your mobile number? This will help me check if you're already registered in our system or if we need to create a new profile for you."
                },
                {
                    "role": "user",
                    "content": "883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to check if this customer exists in our system by looking up their phone number. This will tell me if they're already registered or if I need to create a new customer profile.\n\n<uc_function_call>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"name\":\"datalink.lineagedemo.get_customer_details\",\"arguments\":\"{\\\"Phone_no\\\":\\\"883344\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"1Ub55Lz8A66VmJHgb2K1\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nThank you for providing your phone number. I don't see any existing records associated with the number 883344 in our system. I'd be happy to get you registered so we can proceed with booking your service appointment.\n\nCould you please provide me with the following information:\n1. Your full name\n2. Your email address\n3. Your complete phone number (it seems the one you provided might be partial)\n\nOnce I have these details, I'll create your customer profile in our system."
                },
                {
                    "role": "user",
                    "content": "Varun\nvarun@gmail.com"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for providing your name and email address, Varun. I still need your complete phone number to register you in our system. The number you provided earlier (883344) appears to be partial. \n\nCould you please provide your full phone number, including any country or area codes? This will help us contact you regarding your service appointment if needed."
                },
                {
                    "role": "user",
                    "content": "this is my complete phone num 883344"
                },
                {
                    "role": "assistant",
                    "content": "I need to register the new customer in the system with their provided details: name (Varun), phone number (883344), and email (varun@gmail.com).\n\n<uc_function_call>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"name\":\"datalink.lineagedemo.insert_customer_details\",\"arguments\":\"{\\\"customer_name\\\":\\\"Varun\\\",\\\"phone_number\\\":\\\"883344\\\",\\\"email\\\":\\\"varun@gmail.com\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"HVdLZY7mknvldNEoehiW\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Customer created. ID: 70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\"]]}\"}</uc_function_result>\n\nGreat! I've successfully registered you in our system, Varun. Your customer profile has been created.\n\nNow, to proceed with booking a service appointment, I'll need some information about your vehicle. Could you please provide:\n\n1. Your car model (e.g., Honda Civic, Toyota Camry)\n2. License plate number\n3. Year of manufacture\n\nOnce I have these details, I can register your vehicle and check for available service slots."
                },
                {
                    "role": "user",
                    "content": "Baleno V3\nKL14 \n2023"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.insert_vehicle_details\n\n<uc_function_call>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"name\":\"datalink.lineagedemo.insert_vehicle_details\",\"arguments\":\"{\\\"cutomer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"model\\\":\\\"Baleno V3\\\",\\\"plate_number\\\":\\\"KL14\\\",\\\"year\\\":\\\"2023\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"9Q6dELLkQf9cSG9tI45D\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Vehicle info created. ID: 9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.get_available_service\n\n<uc_function_call>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"name\":\"datalink.lineagedemo.get_available_service\",\"arguments\":\"{}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"rOfwlL9xPMYIZT4JcLjg\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"service_type_id\\\",\\\"service_name\\\",\\\"description\\\"],\\\"rows\\\":[[\\\"S001\\\",\\\"Periodic Maintenance\\\",\\\"Routine maintenance at mileage intervals\\\"],[\\\"S002\\\",\\\"Engine Diagnostics\\\",\\\"Check engine and performance issues\\\"],[\\\"S003\\\",\\\"AC Repair\\\",\\\"Air conditioning service and repair\\\"]]}\"}</uc_function_result>\n\nThank you, Varun! I've successfully registered your 2023 Baleno V3 with plate number KL14 in our system.\n\nNow, let's proceed with booking your service appointment. We offer the following service types:\n\n1. Periodic Maintenance - Routine maintenance at mileage intervals\n2. Engine Diagnostics - Check engine and performance issues\n3. AC Repair - Air conditioning service and repair\n\nWhich type of service would you like to book for your Baleno? Also, do you have a preferred service center location?"
                },
                {
                    "role": "user",
                    "content": "AC Repair"
                },
                {
                    "role": "assistant",
                    "content": "Thank you for selecting AC Repair for your 2023 Baleno V3. \n\nTo proceed with booking your appointment, I'll need to know which service center location you prefer. This will help me check the available time slots at that specific center.\n\nCould you please let me know which service center you'd like to visit? We have multiple locations available."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.get_available_slots\n\n<uc_function_call>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"name\":\"datalink.lineagedemo.get_available_slots\",\"arguments\":\"{\\\"center_name\\\":\\\"Jeddah\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"UTnoAUpZKlp7zHH8bl5i\",\"content\":\"<Empty output>\"}</uc_function_result>\n\nI apologize, but I don't see any service center named \"Jeddah\" in our system, or there might not be any available slots at that location currently.\n\nCould you please choose from one of our other service center locations? Unfortunately, I don't have the complete list of our centers in the system. Common locations might include downtown centers, mall locations, or centers in specific neighborhoods.\n\nAlternatively, if you know another name for the Jeddah center or have a different preferred location, please let me know."
                },
                {
                    "role": "user",
                    "content": "Jeddah"
                },
                {
                    "role": "assistant",
                    "content": "I apologize for the confusion. It seems we're having trouble finding a service center named \"Jeddah\" in our system, or there may not be any available slots at that location currently.\n\nLet me try to help you in a different way. Instead of searching by the center name, could you tell me which general area or city you're located in? This might help us identify the nearest service center to you.\n\nAlternatively, would you like me to check if we have any other service centers with available slots? This would give us a starting point to book your AC Repair service."
                },
                {
                    "role": "user",
                    "content": "recheck once"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.get_available_slots\n\n<uc_function_call>{\"id\":\"A0l4sIhNQTztd61DOxkR\",\"name\":\"datalink.lineagedemo.get_available_slots\",\"arguments\":\"{\\\"center_name\\\":\\\"Jeddah\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"A0l4sIhNQTztd61DOxkR\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"slot_id\\\",\\\"center_id\\\",\\\"location\\\",\\\"slot_datetime\\\",\\\"is_available\\\"],\\\"rows\\\":[[\\\"SL001\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T09:00:00.000+0000\\\",true],[\\\"SL002\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T11:00:00.000+0000\\\",true],[\\\"SL003\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T13:00:00.000+0000\\\",true],[\\\"SL004\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T15:00:00.000+0000\\\",true],[\\\"SL005\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-20T17:00:00.000+0000\\\",true],[\\\"SL016\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T09:00:00.000+0000\\\",true],[\\\"SL017\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T11:00:00.000+0000\\\",true],[\\\"SL018\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T13:00:00.000+0000\\\",true],[\\\"SL019\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T15:00:00.000+0000\\\",true],[\\\"SL020\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-21T17:00:00.000+0000\\\",true],[\\\"SL031\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T09:00:00.000+0000\\\",true],[\\\"SL032\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T11:00:00.000+0000\\\",true],[\\\"SL033\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T13:00:00.000+0000\\\",true],[\\\"SL034\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T15:00:00.000+0000\\\",true],[\\\"SL035\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-22T17:00:00.000+0000\\\",true],[\\\"SL046\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T09:00:00.000+0000\\\",true],[\\\"SL047\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T11:00:00.000+0000\\\",true],[\\\"SL048\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T13:00:00.000+0000\\\",true],[\\\"SL049\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T15:00:00.000+0000\\\",true],[\\\"SL050\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-23T17:00:00.000+0000\\\",true],[\\\"SL061\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T09:00:00.000+0000\\\",true],[\\\"SL062\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T11:00:00.000+0000\\\",true],[\\\"SL063\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T13:00:00.000+0000\\\",true],[\\\"SL064\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T15:00:00.000+0000\\\",true],[\\\"SL065\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-24T17:00:00.000+0000\\\",true],[\\\"SL076\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T09:00:00.000+0000\\\",true],[\\\"SL077\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T11:00:00.000+0000\\\",true],[\\\"SL078\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T13:00:00.000+0000\\\",true],[\\\"SL079\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T15:00:00.000+0000\\\",true],[\\\"SL080\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-25T17:00:00.000+0000\\\",true],[\\\"SL091\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T09:00:00.000+0000\\\",true],[\\\"SL092\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T11:00:00.000+0000\\\",true],[\\\"SL093\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T13:00:00.000+0000\\\",true],[\\\"SL094\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T15:00:00.000+0000\\\",true],[\\\"SL095\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-26T17:00:00.000+0000\\\",true],[\\\"SL106\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T09:00:00.000+0000\\\",true],[\\\"SL107\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T11:00:00.000+0000\\\",true],[\\\"SL108\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T13:00:00.000+0000\\\",true],[\\\"SL109\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T15:00:00.000+0000\\\",true],[\\\"SL110\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-27T17:00:00.000+0000\\\",true],[\\\"SL121\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T09:00:00.000+0000\\\",true],[\\\"SL122\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T11:00:00.000+0000\\\",true],[\\\"SL123\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T13:00:00.000+0000\\\",true],[\\\"SL124\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T15:00:00.000+0000\\\",true],[\\\"SL125\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-28T17:00:00.000+0000\\\",true],[\\\"SL136\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T09:00:00.000+0000\\\",true],[\\\"SL137\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T11:00:00.000+0000\\\",true],[\\\"SL138\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T13:00:00.000+0000\\\",true],[\\\"SL139\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T15:00:00.000+0000\\\",true],[\\\"SL140\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-29T17:00:00.000+0000\\\",true],[\\\"SL151\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T09:00:00.000+0000\\\",true],[\\\"SL152\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T11:00:00.000+0000\\\",true],[\\\"SL153\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T13:00:00.000+0000\\\",true],[\\\"SL154\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T15:00:00.000+0000\\\",true],[\\\"SL155\\\",\\\"SC001\\\",\\\"Jeddah - City Road\\\",\\\"2025-05-30T17:00:00.000+0000\\\",true]]}\"}</uc_function_result>\n\nThank you for your patience! I've successfully found available slots at our Jeddah - City Road service center. Here are some of the available appointment times:\n\nFor May 20, 2025:\n- 9:00 AM\n- 11:00 AM\n- 1:00 PM\n- 3:00 PM\n- 5:00 PM\n\nFor May 21, 2025:\n- 9:00 AM\n- 11:00 AM\n- 1:00 PM\n- 3:00 PM\n- 5:00 PM\n\nWe also have similar time slots available on subsequent days through May 30, 2025.\n\nWhich date and time would you prefer for your AC Repair service appointment?"
                },
                {
                    "role": "user",
                    "content": "Book for May 20 9:00 AM"
                },
                {
                    "role": "assistant",
                    "content": "I'm going to use datalink.lineagedemo.process_booking_status\n\n<uc_function_call>{\"id\":\"QIIdm7pwmPVkYk9ClFDE\",\"name\":\"datalink.lineagedemo.process_booking_status\",\"arguments\":\"{\\\"operation_type\\\":\\\"CREATE\\\",\\\"customer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"old_slot_id\\\":\\\"NULL\\\",\\\"slot_id\\\":\\\"SL001\\\",\\\"service_type_id\\\":\\\"S003\\\",\\\"vehicle_id\\\":\\\"9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\",\\\"center_id\\\":\\\"SC001\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"QIIdm7pwmPVkYk9ClFDE\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Invalid operation type: CREATE. Use INSERT, UPDATE, or DELETE.\\\"]]}\"}</uc_function_result>\n\nI'm going to use datalink.lineagedemo.process_booking_status\n\n<uc_function_call>{\"id\":\"QUUgyR2D6dDAuU0OX5Ex\",\"name\":\"datalink.lineagedemo.process_booking_status\",\"arguments\":\"{\\\"operation_type\\\":\\\"INSERT\\\",\\\"customer_id\\\":\\\"70bda82c-3fab-4b6e-b4e6-46477e7d7b48\\\",\\\"old_slot_id\\\":\\\"NULL\\\",\\\"slot_id\\\":\\\"SL001\\\",\\\"service_type_id\\\":\\\"S003\\\",\\\"vehicle_id\\\":\\\"9231be27-cdd8-4262-ab4f-5c31c8ad1408\\\",\\\"center_id\\\":\\\"SC001\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"QUUgyR2D6dDAuU0OX5Ex\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"Booking created. ID: fec3b58f-fdf7-4cc5-a311-161f1b542b57, Status: CONFIRMED\\\"]]}\"}</uc_function_result>\n\nGreat news, Varun! I've successfully booked your AC Repair service appointment for May 20, 2025, at 9:00 AM at our Jeddah - City Road service center.\n\nYour booking details:\n- Service: AC Repair\n- Date and Time: May 20, 2025, at 9:00 AM\n- Location: Jeddah - City Road service center\n- Vehicle: 2023 Baleno V3 (KL14)\n- Booking Status: CONFIRMED\n- Booking ID: fec3b58f-fdf7-4cc5-a311-161f1b542b57\n\nWe recommend arriving about 10-15 minutes before your appointment time. Our technicians will inspect your air conditioning system and perform the necessary repairs.\n\nIs there anything else you would like to know about your upcoming service appointment?"
                },
                {
                    "role": "user",
                    "content": "Hey i want to  reschedule"
                }
            ]
        },
        "expected_response": None
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)


# COMMAND ----------

import mlflow

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "datalink"
schema = "lineagedemo"
model_name = "alj_cc_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/deploy-agent) for details
