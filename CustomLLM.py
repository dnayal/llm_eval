# evaluate NVIDIA & Llama 1B models
# use custom evaluation
# use deepeval for evaluation
# use benchmark scores
import os
from deepeval.models import DeepEvalBaseLLM
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

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
    #model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self):
        model = AutoModelForCausalLM.from_pretrained(CustomLlamaLLM.model_name)
        tokenizer = AutoTokenizer.from_pretrained(CustomLlamaLLM.model_name)
        pipe = pipeline(
            task = "text-generation",
            model = model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=500,
#            num_return_sequences=1,
#            eos_token_id=tokenizer.eos_token_id,
#            pad_token_id=tokenizer.eos_token_id,
            device_map="auto"
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        template = """
        Question: {question} 
        Instruction: Give specific and precise answers. Do not include anything in your response that does not relate to the question asked.
        Answer: 
        """
        prompt = PromptTemplate.from_template(template)
        self.chain =  prompt | llm | (lambda output: output.split("Answer:")[-1].strip())
        

    def load_model(self):
        return self.chain


    def generate(self, prompt: str) -> str:
        response = self.chain.invoke({"question": prompt})
        return response

    async def a_generate(self, prompt: str) -> str:
        response = await self.chain.ainvoke({"question": prompt})
        return response
        
    def get_model_name(self):
        return CustomLlamaLLM.get_model_name        