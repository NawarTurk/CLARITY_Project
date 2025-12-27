import json
from pathlib import Path
import re

AGENT_EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "datasets" / "agent_examples" / "v2"

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
        return "\n\n".join(blocks) # ++

    def _build_prompt(self):       
        """Build agent's system prompt with examples."""

        return f""" 
        You are a world-class political discourse analyst trained to detect evasive communication strategies in high-stakes interviews.

        Your specialization: {self.target_label}

        Your task:
        Given a journalist’s question and a politician’s answer, decide whether the answer belongs to your specialization.

       Rating rule:
        - Output ONLY a number from 0-100
        - 0 = definitely NOT {self.target_label}
        - 50 = uncertain
        - 100 = definitely IS {self.target_label}

        Below are examples that BELONG to your specialization:
        { self._format_examples(self.positive_examples)}

        Below are examples that DO NOT belong to your specialization:
        {self._format_examples(self.negative_examples)}
        """.strip()

    def predict(self, question, answer):        
        """Return confidence score 0-100."""
        prompt = (
            self.prompt + 
            f"""
            Now judge the following case:
            Q: {question}
            A: {answer}
            Confidence (0-100):
            """
        )

        prediction = self.llm_call(prompt).strip()
        
        # extract first number between 0–100
        match = re.search(r"\b\d{1,3}\b", prediction)
        if not match:
            return 50  # fallback = uncertain

        score = int(match.group())
        print( f"Raw score prediction for '{self.target_label}': {score}" )

        return max(0, min(score, 100))
        
    def explain(self, question, answer, predicted_score):
        """Return explanation for YES prediction."""
        
        prompt = (
            self.prompt + 
            f"""
            Given this case:
            Q: {question}
            A: {answer}

            You rated this {predicted_score}/100 confidence it belongs to {self.target_label}.
            Briefly explain (1-2 sentences) why:
    
            """)

        return self.llm_call(prompt).strip()


 