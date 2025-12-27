import json
from pathlib import Path

AGENT_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "agent_examples"

class Agent:
    """Specialist agent for one clarity class (one-vs-rest)."""

    def __init__(self, target_label, llm_call):
        agent_examples_path =  AGENT_EXAMPLES_DIR / f"agent_examples_{target_label.replace(' ', '_').lower()}.json"

        with open(agent_examples_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.target_label =  data['target_label']
        self.positive_examples = data['positives']
        self.negative_examples = data['negatives']
        self.prompt = self._build_prompt()
        self.llm_call = llm_call

    def _format_examples(self, examples):
        """Format Q&A examples as readable text blocks."""

        blocks = []
        for ex in examples:
            blocks.append(f"Q: {ex['question']}\nA: {ex['answer']}\n")
        return "\n".join(blocks)

    def _build_prompt(self):       
        """Build agent's system prompt with examples."""

        return f""" 
        You are a world-class political discourse analyst trained to detect evasive communication strategies in high-stakes interviews.

        Your specialization: {self.target_label}

        Your task:
        Given a journalist’s question and a politician’s answer, decide whether the answer belongs to your specialization.

        Decision rule:
        - Answer ONLY with YES or NO
        - YES means the answer belongs to {self.target_label}
        - NO means it does not

        Below are examples that BELONG to your specialization:
        { self._format_examples(self.positive_examples)}

        Below are examples that DO NOT belong to your specialization:
        {self._format_examples(self.negative_examples)}
        """.strip()

    def predict(self, question, answer):        
        """Return YES or NO only."""

        prompt = (
            self.prompt + 
            f"""
            Now judge the following case:
            Q: {question}
            A: {answer}
            Answer
            """
        )

        prediction = self.llm_call(prompt).strip().upper()

        return "YES" if prediction.startswith("YES") else "NO"
    
    def explain(self, question, answer):
        """Return explanation for YES prediction."""
        
        prompt = (
            self.prompt + 
            f"""
            Now judge the following case:
            Q: {question}
            A: {answer}
            Answer: YES

            You answered YES. Explain why the answer belongs to {self.target_label}.
            """.strip()
        )

        return self.llm_call(prompt).strip()


 