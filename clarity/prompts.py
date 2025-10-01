from pathlib import Path


def load_template(prompts_dir: Path, template_name: str) -> str:
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def build_prompt(template: str, *, question: str, answer: str) -> str:
    return template.format(question=question, answer=answer)
