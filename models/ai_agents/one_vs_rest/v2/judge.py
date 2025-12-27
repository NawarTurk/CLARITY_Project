class Judge:
    """Judge to evaluate responses from one-vs-rest agents."""

    def __init__(self, llm_call):
        self.llm_call = llm_call
    
    def judge(self, question, answer, rationale):
        """Evaluate agent votes and provide final decision with rationale."""
        
        labels = ",".join(rationale.keys())

        prompt = f"""
        You are a world-class political discourse analyst specializing in CLARITY classification.
        Your task: Resolve close-confidence agent predictions (confidence margin below threshold) between {labels} to assign a single CLARITY label.

        Question: {question}
        Answer: {answer}

        Agents' Rationales: {"\n".join([f"{k}:{v}" for k,v in rationale.items()])}

        Focus: Distinguish between {labels} based on the agents' rationales.
        Choose ONE final CLARITY label from: {labels}

        ONLY OUTPUT THE final CLARITY label:

        Label:
        """
        return self.llm_call(prompt.strip())
