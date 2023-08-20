from langchain import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

region = "us-east-1"
endpoint_name = "huggingface-pytorch-tgi-inference-2023-08-20-10-04-46-379"
DB_FAISS_PATH = 'vectorstore/db_faiss'

def build_chain():

    # Sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cpu'})

    # Laod Faiss index
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    # Default system prompt for the LLama2
    system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    # Custom ContentHandler to handle input and output to the SageMaker Endpoint
    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "do_sample": True,
                    "top_p": 0.6,
                    "temperature": 0.8,
                    "top_k": 50,
                    "max_new_tokens": 512,
                    "repetition_penalty": 1.03,
                    "stop": ["</s>"]
                }
            }
            input_str = json.dumps(
                payload,
            )
            return input_str.encode("utf-8")

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            content = response_json[0]["generated_text"]
            return content

    # Langchain chain for invoking SageMaker Endpoint
    llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=region,
        content_handler=ContentHandler(),
        callbacks=[StreamingStdOutCallbackHandler()],
        endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    )


    def get_chat_history(inputs) -> str:
        res = []
        for _i in inputs:
            if _i.get("role") == "user":
                user_content = _i.get("content")
            if _i.get("role") == "assistant":
                assistant_content = _i.get("content")
                res.append(f"user:{user_content}\nassistant:{assistant_content}")
        return "\n".join(res)

    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(
        condense_qa_template,
    )
    
    # Langchain chain for Conversation
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        get_chat_history=get_chat_history
    )
    return qa


def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})

chain  = build_chain()
prompt = "what is Cuterebra?"

result = run_chain(chain=chain, prompt=prompt)

print("Result: ", result)


