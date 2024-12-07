import json
import os
from deepeval.models import DeepEvalBaseLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

# custom model for Nvidia LLM
class CustomNvidiaLLM(DeepEvalBaseLLM):
    
    model_name = "meta/llama-3.1-405b-instruct"

    def __init__(self):
        os.environ["NVIDIA_API_KEY"] = "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"
        model = ChatNVIDIA(model = CustomNvidiaLLM.model_name)
        self.model = model
        
    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        return model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()
        res = await model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return CustomNvidiaLLM.model_name


# custom model for Llama
class CustomLlamaLLM(DeepEvalBaseLLM):

    model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self):
        model = AutoModelForCausalLM.from_pretrained(
            CustomLlamaLLM.model_name,
            device_map = "auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(CustomLlamaLLM.model_name)
        self.model = model
        self.tokenizer = tokenizer


    def load_model(self):
        return self.model


    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()

        pipe = pipeline(
            "text-generation",
            model = model,
            tokenizer = self.tokenizer,
            use_cache = True,
            device_map = "auto",
            max_new_tokens = 500,
            do_sample = True,
            top_k = 5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Create parser required for JSON confinement using lmformatenforcer
        # parser = JsonSchemaParser(schema.schema())

        # CHECK: Comment the following line and uncomment the previous one if you get errors with parsing the schema
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipe.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipe(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt) :]
        
        # CHECK: Uncomment following line if you get errors
        # For some weird reason printing the JSON output does not throw any error
        # print("==> Raw Output:", output)  # Debug the raw output

        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return CustomLlamaLLM.get_model_name        