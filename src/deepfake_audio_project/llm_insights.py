import os
from pathlib import Path

import requests


def _build_prompt(test_metrics):
    return f"""
You are an ML reviewer helping prepare a journal-ready deepfake audio detection paper.
Given the evaluation metrics below, provide:
1) strengths
2) weaknesses / threats to validity
3) experiments to add before submission
4) concrete ablation study plan
5) model improvement recommendations
6) a short publication-style summary paragraph

Metrics:
- Samples: {test_metrics.get('n_samples')}
- Accuracy: {test_metrics.get('accuracy')}
- Accuracy CI95: {test_metrics.get('accuracy_ci_95')}
- Balanced accuracy: {test_metrics.get('balanced_accuracy')}
- Precision: {test_metrics.get('precision')}
- Recall: {test_metrics.get('recall')}
- F1: {test_metrics.get('f1')}
- F1 CI95: {test_metrics.get('f1_ci_95_bootstrap')}
- ROC-AUC: {test_metrics.get('roc_auc')}
- PR-AUC: {test_metrics.get('pr_auc')}
- MCC: {test_metrics.get('mcc')}
- Brier score: {test_metrics.get('brier_score')}
- Log-loss: {test_metrics.get('log_loss')}
- ECE: {test_metrics.get('ece')}
- Default threshold: {test_metrics.get('threshold_default')}
- Optimal F1 threshold: {test_metrics.get('threshold_optimal_f1')}
- F1 at optimal threshold: {test_metrics.get('f1_at_optimal_threshold')}

Format the response in markdown with clear section titles.
""".strip()


def generate_llm_review(test_metrics, save_dir="outputs", model_name="gpt-4o-mini"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Skipping LLM insights generation.")
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a rigorous ML research assistant."},
            {"role": "user", "content": _build_prompt(test_metrics)},
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"LLM insight generation failed: {exc}")
        return None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / "llm_research_review.md"
    out_file.write_text(content, encoding="utf-8")
    print(f"LLM review saved to: {out_file}")
    return out_file

