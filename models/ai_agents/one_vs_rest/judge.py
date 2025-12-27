class Judge:
    """Judge to evaluate responses from one-vs-rest agents."""

    def __init__(self, llm_call):
        self.llm_call = llm_call
    
    def judge(self, question, answer, votes, rationale):
        """Evaluate agent votes and provide final decision with rationale."""
        
        yes_labels = ",".join([k for k, v in votes.items() if v == "YES"])

        prompt = f"""
        You are a world-class political discourse analyst specializing in CLARITY classification.
        Your task: Resolve conflicting agent votes between {yes_labels} to assign a single CLARITY label.
        Question: {question}
        Answer: {answer}
        Agents' Rationales: {"\n".join([f"{k}:{v}" for k,v in rationale.items()])}
        Focus: Distinguish between {yes_labels} based on the agents' rationales.
        Choose ONE final CLARITY label from: {yes_labels}

        ONLY OUTPUT THE final CLARITY label:

        Label:
        """
        return self.llm_call(prompt.strip())
