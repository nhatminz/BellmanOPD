"""MMLU-Pro's local parquet schema, zero-shot prompt and answer grading.

Reference schema: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
This is sampled zero-shot CoT, not the official few-shot greedy leaderboard.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PROTOCOL = "mmlu_pro_zero_shot_cot_v1"
LETTERS = "ABCDEFGHIJ"
INSTRUCTION = (
    "The following is a multiple-choice question about {category}. "
    "Choose the single best answer. Think step by step, then end your response "
    'with exactly "The answer is (X).", where X is the letter of your choice.'
)


def load_questions(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    table = pq.read_table(path)
    required = {"question", "options", "category"}
    if not required <= set(table.column_names) or not (
        {"answer", "answer_index"} & set(table.column_names)
    ):
        raise ValueError(
            "MMLU-Pro parquet requires question, options, category and "
            f"answer or answer_index; found {table.column_names}"
        )
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    questions = []
    seen = set()
    for index, row in enumerate(rows):
        question, category, options = row["question"], row["category"], row["options"]
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(category, str)
            or not category.strip()
            or not isinstance(options, list)
            or not 2 <= len(options) <= len(LETTERS)
            or any(
                not isinstance(option, str) or not option.strip() for option in options
            )
        ):
            raise ValueError(
                f"Invalid MMLU-Pro question/options/category at row {index}"
            )
        answer = row.get("answer")
        answer = str(answer).strip().upper() if answer is not None else None
        answer_index = row.get("answer_index")
        if answer_index is not None:
            if (
                isinstance(answer_index, bool)
                or not isinstance(answer_index, int)
                or not 0 <= answer_index < len(options)
            ):
                raise ValueError(
                    f"Invalid answer_index at row {index}: {answer_index!r}"
                )
            indexed_answer = LETTERS[answer_index]
            if answer is not None and answer != indexed_answer:
                raise ValueError(f"answer/answer_index disagree at row {index}")
            answer = indexed_answer
        if (
            not isinstance(answer, str)
            or len(answer) != 1
            or answer not in LETTERS[: len(options)]
        ):
            raise ValueError(f"Invalid reference answer at row {index}: {answer!r}")
        identifier = str(row.get("question_id", index))
        if identifier in seen:
            raise ValueError(f"Duplicate question_id: {identifier}")
        seen.add(identifier)
        # Deliberately exclude cot_content and all other reference explanations.
        questions.append(
            {
                "index": index,
                "question_id": identifier,
                "question": question,
                "options": options,
                "category": category,
                "answer": answer,
            }
        )
    if not questions:
        raise ValueError("MMLU-Pro dataset is empty")
    return questions


def question_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    choices = "\n".join(
        f"{LETTERS[i]}. {option}" for i, option in enumerate(row["options"])
    )
    return [
        {
            "role": "user",
            "content": (
                INSTRUCTION.format(category=row["category"])
                + f"\n\nQuestion:\n{row['question']}\n\nOptions:\n{choices}\n\nAnswer:"
            ),
        }
    ]


def extract_answer(response: str, num_options: int) -> str | None:
    """Require an explicit final answer; never guess from letters in reasoning."""
    # Qwen's reasoning block can mention/reject options. Grade only the final
    # response after a closing think tag, when the tag is present.
    text = response.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return None  # an unfinished reasoning response has no final answer
    text = text.replace("**", "").replace("`", "")
    patterns = (
        r"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*\(?([A-J])\)?(?!\w)",
        r"\\boxed\s*\{\s*\(?([A-J])\)?\s*\}",
    )
    matches = []
    for pattern in patterns:
        matches.extend(
            (match.start(), match.group(1).upper())
            for match in re.finditer(pattern, text, re.I)
        )
    if matches:
        answer = max(matches)[1]
    else:
        match = re.fullmatch(r"\(?([A-J])\)?[.!]?", text, re.I)
        answer = match.group(1).upper() if match else None
    return answer if answer in list(LETTERS[:num_options]) else None


def score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot report MMLU-Pro metrics on zero questions")
    count = len(rows)
    correct = invalid = truncated = 0
    solved = 0
    for row in rows:
        responses = row["responses"]
        if len(responses) != 8:
            raise ValueError(
                f"Expected exactly eight responses for question {row['question_id']}"
            )
        grades = []
        for response in responses:
            parsed = extract_answer(response["text"], len(row["options"]))
            grade = parsed == row["answer"]
            if (
                response.get("parsed_answer") != parsed
                or response.get("correct") != grade
            ):
                raise ValueError(
                    f"Inconsistent response grade for question {row['question_id']}"
                )
            invalid += parsed is None
            truncated += response.get("finish_reason") == "length"
            grades.append(grade)
        correct += sum(grades)
        solved += any(grades)
    return {
        "questions": count,
        "completions": count * 8,
        "correct_completions": correct,
        "solved_questions": solved,
        "avg_at_8": correct / (count * 8),
        "pass_at_8": solved / count,
        "invalid_answer_rate": invalid / (count * 8),
        "truncated_response_rate": truncated / (count * 8),
    }


def metrics_by_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    return {name: score_predictions(values) for name, values in sorted(groups.items())}


def aggregate_seeds(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) < 2 or len({row["seed"] for row in results}) != len(results):
        raise ValueError(
            "At least two distinct seeds are required for sample standard deviation"
        )
    if len({row["overall"]["questions"] for row in results}) != 1:
        raise ValueError("Seed results contain different question counts")
    categories = tuple(results[0]["by_category"])
    if any(tuple(row["by_category"]) != categories for row in results):
        raise ValueError("Seed results contain different categories")

    def aggregate(values):
        if len({value["questions"] for value in values}) != 1:
            raise ValueError("Seed results contain different category question counts")
        summary = {"questions": values[0]["questions"], "num_seeds": len(values)}
        for key in (
            "avg_at_8",
            "pass_at_8",
            "invalid_answer_rate",
            "truncated_response_rate",
        ):
            samples = [float(value[key]) for value in values]
            mean, std = statistics.mean(samples), statistics.stdev(samples)
            summary[key] = {
                "values": samples,
                "mean": mean,
                "std": std,
                "std_ddof": 1,
                "percent_mean": mean * 100,
                "percent_std": std * 100,
                "formatted_percent": f"{mean * 100:.2f} ± {std * 100:.2f}",
            }
        return summary

    return {
        "seeds": [row["seed"] for row in results],
        "per_seed": results,
        "overall": aggregate([row["overall"] for row in results]),
        "by_category": {
            name: aggregate([row["by_category"][name] for row in results])
            for name in categories
        },
    }
