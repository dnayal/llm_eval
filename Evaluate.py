from CustomLLM import CustomLlamaLLM, CustomNvidiaLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.benchmarks import MMLU, HellaSwag, BigBenchHard
from deepeval.benchmarks.tasks import MMLUTask, HellaSwagTask, BigBenchHardTask

def LLMTestCaseWrapper(question, model):
    response = model.generate(question)
    llmtestcase = LLMTestCase(
        input = question,
        actual_output = response,
        context = ["Just give me the name of the president - do not give long explainations."]
    )
    return llmtestcase

    
"""
customllama = CustomLlamaLLM()
llama_relevancy_metric = AnswerRelevancyMetric(
    model=customllama
)
llama_relevancy_metric.measure(LLMTestCaseWrapper(question, customllama))
print(f"[Relevancy] Score: {llama_relevancy_metric.score} Reason: {llama_relevancy_metric.reason}")
"""

customnvidia = CustomNvidiaLLM()
scores = {}

# Relevancy Metric
question = "who is the president of USA?"
nvidia_relevancy_metric = AnswerRelevancyMetric(
    model = customnvidia,
    threshold = 0.7
)
nvidia_relevancy_metric.measure(LLMTestCaseWrapper(question, customnvidia))
scores["Relevancy"] = nvidia_relevancy_metric.score

# MMLU Metric
nvidia_mmlu_metric = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots = 5
)
nvidia_mmlu_metric.evaluate(model = customnvidia)
scores["MMLU"] = nvidia_mmlu_metric.overall_score

# HellaSwag Metric
nvidia_hellaswag_metric = HellaSwag(
    tasks = [HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING, HellaSwagTask.MAKING_A_SANDWICH],
    n_shots = 5
)
nvidia_hellaswag_metric.evaluate(model = customnvidia)
scores["HellaSwag"] = nvidia_hellaswag_metric.overall_score

# Big Bench Hard
nvidia_bigbench_metric = BigBenchHard(
    tasks = [BigBenchHardTask.BOOLEAN_EXPRESSIONS, BigBenchHardTask.CAUSAL_JUDGEMENT, BigBenchHardTask.REASONING_ABOUT_COLORED_OBJECTS],
    n_shots = 3,
    enable_cot =  True
)
nvidia_bigbench_metric.evaluate(model = customnvidia)
scores["Big Bench Hard"] = nvidia_bigbench_metric.overall_score

print("Evaluation Scores for ", CustomNvidiaLLM.get_model_name())
for metric, score in scores:
    print(f"[{metric}] : {score}")