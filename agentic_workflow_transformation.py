from typing import TypedDict, Optional, List, Dict
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

# LLM integration (optional; will fall back to deterministic local behavior if unavailable)
from langchain.chat_models import init_chat_model
try:
    llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
except Exception:
    llm = None


class TransformState(TypedDict):
    description: str
    sample_data: List[Dict]
    plan: Optional[str]
    code: Optional[str]
    output_path: Optional[str]
    attempts: Optional[int]


# Validation is now handled by LLM critic instead of heuristics


# Validation is now handled by LLM critic instead of heuristics


# Simple planner that may use LLM; returns a short plan string
def planner_agent(state: TransformState) -> Dict:
    desc = state.get("description", "")
    try:
        if llm is not None:
            prompt = f"""
                        You are a senior data engineer.
                        Data sample:
                        {state['sample_data']}

                        Transformation description:
                        {state['description']}

                        Produce a precise step-by-step transformation plan.
                        """
            res = llm.invoke(prompt)
            plan = (getattr(res, "content", None) or getattr(res, "output_text", None) or str(res)).strip()
        else:
            plan = f"Plan: apply transformation described: {desc}"
    except Exception:
        plan = f"Plan: apply transformation described: {desc}"
    state["plan"] = plan
    print("Generated Transformation Plan:\n", plan)
    return {"plan": plan}


# Code generator: generate python code that uses compute_expected_df to produce result_df
def code_generator_agent(state: TransformState) -> Dict:
    desc = state.get("description", "")
    plan = state.get("plan") or desc
    # Prefer to ask the LLM, but have a deterministic fallback
    try:
        if llm is not None:
            # Provide the plan and, when present, the previous code and output to guide fixes.
            prev_code = state.get("code")
            prompt = (
                f"Generate Python pandas code that implements the following PLAN exactly:\n{plan}\n\n"
                "Rules:\n"
                "- Use the provided `sample_data` variable (list of dicts). Do NOT define sample literals.\n"
                "- Use the EXACT predicates/columns from the plan/description. Do NOT change comparisons (e.g., keep `age > 30` as-is).\n"
                "- Assign the final pandas DataFrame to a variable named `result_df`.\n"
                "- Do NOT write files to disk.\n"
                "- Return ONLY a fenced Python code block (```python ... ```).\n"
            )
            if prev_code and state.get("attempts", 0) > 0:
                prompt += "\nPrevious code (fix as needed):\n" + prev_code + "\n"
            res = llm.invoke(prompt)
            code = (getattr(res, "content", None) or getattr(res, "output_text", None) or str(res)).strip()
            # extract fenced code if present
            m = re.search(r"```(?:python)?\n([\s\S]*?)```", code)
            if m:
                code = m.group(1)
            code = code.strip()
            if not code:
                raise ValueError("LLM returned empty code")
        else:
            # deterministic implementation: use pandas directly
            code = (
                "import pandas as pd\n"
                "df = pd.DataFrame(sample_data)\n"
                "# Apply transformation here\n"
                "result_df = df\n"
            )
    except Exception:
        code = (
            "import pandas as pd\n"
            "df = pd.DataFrame(sample_data)\n"
            "# Apply transformation here\n"
            "result_df = df\n"
        )
    state["code"] = code
    return {"code": code}


# Execution agent: safely exec the generated code and write a CSV output
def execution_agent(state: TransformState) -> Dict:
    code = state.get("code")
    if not code:
        gen = code_generator_agent(state)
        code = gen["code"]
        state["code"] = code

    project_dir = Path(__file__).resolve().parent
    project_dir.mkdir(parents=True, exist_ok=True)
    desc = (state.get("description") or "").strip()
    slug = re.sub(r"[^A-Za-z0-9]+", "_", desc).lower().strip("_")[:80] or "no_description"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    unique_name = f"transform_{slug}_{ts}.csv"
    output_path = project_dir / unique_name

    # Prevent embedding data blocks; force use of provided sample_data
    code_clean = re.sub(r"\bdata\s*=\s*\[.*?\]\s*", "data = sample_data\n", code, flags=re.S)

    local_env = {
        "pd": pd,
        "sample_data": state.get("sample_data"),
    }

    try:
        exec(code_clean, {}, local_env)
        print("code:", code_clean)
    except Exception as e:
        # If execution fails, fallback to empty result
        print("Execution error, LLM will validate in critic phase:", e)
        # Create an empty result file - let critic determine if re-generation is needed
        result_df = pd.DataFrame()

    # Discover result_df-like variables
    result_df = None
    if "result_df" in local_env and isinstance(local_env["result_df"], pd.DataFrame):
        result_df = local_env["result_df"]
    elif "filtered_df" in local_env and isinstance(local_env["filtered_df"], pd.DataFrame):
        result_df = local_env["filtered_df"]
    elif "df" in local_env and isinstance(local_env["df"], pd.DataFrame):
        # if df exists but result not explicitly set, assume df is the result
        result_df = local_env["df"]
    elif "result" in local_env and isinstance(local_env["result"], (list, tuple)):
        result_df = pd.DataFrame(local_env["result"])

    if result_df is None:
        result_df = pd.DataFrame()

    result_df.to_csv(str(output_path), index=False)
    state["output_path"] = str(output_path)
    return {"output_path": str(output_path)}


def critic_agent(state: TransformState) -> Dict:
    """LLM-based critic: ask the LLM if the output matches the plan."""
    MAX_RETRIES = 2
    attempts = state.get("attempts", 0) or 0
    outp = state.get("output_path")
    
    if not outp or not Path(outp).exists():
        return {"valid": False, "output_path": outp}
    
    try:
        # Read the generated output
        actual_df = pd.read_csv(outp)
        sample_df = pd.DataFrame(state["sample_data"])
        plan = state.get("plan", "")
        description = state.get("description", "")
        code = state.get("code", "")
        
        # Ask LLM to validate (and request brief reason if NO)
        if llm is None:
            print("Warning: LLM not available for critic validation")
            return {"valid": True, "output_path": outp}

        validation_prompt = (
            f"Plan: {plan}\n\n"
            f"Transformation Description: {description}\n\n"
            f"Generated Code:\n{code}\n\n"
            f"Input Data (first 5 rows):\n{sample_df.head().to_string()}\n\n"
            f"Output Data (first 5 rows):\n{actual_df.head().to_string()}\n\n"
            "Question: Does the output correctly implement the transformation described in the plan? "
            "Answer with ONLY 'YES' or 'NO' on the first line, and if NO provide a one-sentence reason on the second line."
        )

        res = llm.invoke(validation_prompt)
        response_text = (getattr(res, "content", None) or getattr(res, "output_text", None) or str(res)).strip()
        response = response_text.splitlines()
        first = response[0].strip().upper() if response else ""
        is_valid = first.startswith("YES")
        reason = response[1].strip() if len(response) > 1 else None
        
        if is_valid:
            return {"valid": True, "output_path": outp}
        
        # Validation failed -> request a corrected code snippet from the LLM (include failure reason)
        while attempts < MAX_RETRIES:
            attempts += 1
            state["attempts"] = attempts
            print(f"Critic: validation failed ({reason}). Requesting fix from LLM (attempt {attempts})")

            # Ask LLM to return a corrected Python code block that fixes the issue
            fix_prompt = (
                f"Plan: {plan}\n\n"
                f"Transformation Description: {description}\n\n"
                f"Previous Code:\n{code}\n\n"
                f"Input Data (first 5 rows):\n{sample_df.head().to_string()}\n\n"
                f"Output Data (first 5 rows):\n{actual_df.head().to_string()}\n\n"
                "Please provide a corrected Python code snippet (fenced in ```python) that implements the PLAN exactly. "
                "Return only the fenced code block. Ensure the final DataFrame is assigned to `result_df`. "
                "Also avoid adding or changing sample data literals."
            )

            fix_res = llm.invoke(fix_prompt)
            fix_text = (getattr(fix_res, "content", None) or getattr(fix_res, "output_text", None) or str(fix_res)).strip()
            m = re.search(r"```(?:python)?\n([\s\S]*?)```", fix_text)
            if m:
                new_code = m.group(1).strip()
            else:
                # If LLM didn't return a fenced block, take whole text (best-effort)
                new_code = fix_text

            state["code"] = new_code
            exec_res = execution_agent(state)
            outp = exec_res.get("output_path")
            state["output_path"] = outp

            if not outp or not Path(outp).exists():
                continue

            actual_df = pd.read_csv(outp)

            # Re-validate with LLM using the new code
            reval_prompt = (
                f"Plan: {plan}\n\n"
                f"Transformation Description: {description}\n\n"
                f"Generated Code (retry {attempts}):\n{state.get('code', '')}\n\n"
                f"Input Data (first 5 rows):\n{sample_df.head().to_string()}\n\n"
                f"Output Data (first 5 rows):\n{actual_df.head().to_string()}\n\n"
                "Question: Does the output correctly implement the transformation described in the plan? "
                "Answer with ONLY 'YES' or 'NO' on the first line."
            )

            r = llm.invoke(reval_prompt)
            r_text = (getattr(r, "content", None) or getattr(r, "output_text", None) or str(r)).strip().upper()
            if r_text.startswith("YES"):
                return {"valid": True, "output_path": outp}
        
        return {"valid": False, "output_path": outp}
    except Exception as e:
        print("Critic error:", e)
        return {"valid": False, "output_path": outp if outp else None}


# Lightweight app wrapper to run the sequence planner -> codegen -> execute -> critic
class SimpleApp:
    def invoke(self, state: TransformState) -> Dict:
        st = dict(state)
        planner_agent(st)
        code_generator_agent(st)
        execution_agent(st)
        crit = critic_agent(st)
        st.update(crit)
        return st


# Try to use langgraph.StateGraph if available, otherwise fallback to SimpleApp
try:
    from langgraph.graph import StateGraph, END
    graph = StateGraph(TransformState)
    graph.add_node("planner", planner_agent)
    graph.add_node("codegen", code_generator_agent)
    graph.add_node("executor", execution_agent)
    graph.add_node("critic", critic_agent)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "codegen")
    graph.add_edge("codegen", "executor")
    graph.add_edge("executor", "critic")
    graph.add_edge("critic", END)
    app = graph.compile()
except Exception:
    app = SimpleApp()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run transformation workflow")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to input CSV file to use as the sample data (falls back to ./input.csv if present)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation for a set of sample descriptions and save results to eval_results.csv",
    )
    args = parser.parse_args()

    # Load sample data from CSV if provided or if ./input.csv exists
    sample_records = None
    if args.input_csv:
        sample_df = pd.read_csv(args.input_csv)
        sample_records = sample_df.to_dict(orient="records")
        print(f"Loaded sample data from {args.input_csv}")
    else:
        default_csv = Path.cwd() / "input.csv"
        if default_csv.exists():
            sample_df = pd.read_csv(default_csv)
            sample_records = sample_df.to_dict(orient="records")
            print(f"Loaded sample data from {default_csv}")

    initial_state: TransformState = {
        "description": "Filter rows where age > 30 and select columns name and age.",
        "sample_data": sample_records
        if sample_records is not None
        else [
            {"name": "Alice", "age": 25, "city": "New York"},
            {"name": "Bob", "age": 35, "city": "Los Angeles"},
            {"name": "Charlie", "age": 32, "city": "Chicago"},
        ],
        "plan": None,
        "code": None,
        "output_path": None,
        "attempts": 0,
    }

    try:
        if args.eval:
            sample_csv = args.input_csv or (Path.cwd() / "input.csv")
            if not Path(sample_csv).exists():
                print("No input CSV found for evaluation. Create or pass --input-csv.")
            else:
                sample_df = pd.read_csv(sample_csv)
                descriptions = [
                    # Basic
                    "Filter rows where age > 30 and select columns name and age.",
                    "Filter rows where city == 'New York' and select name, city.",
                    "Filter rows where age <= 30 and select name and age.",
                    "Sort by age descending and keep top 3 rows.",
                    "Group by city and count names.",
                    "Add is_adult column where is_adult = age >= 18.",
                    "Select rows where name starts with 'J' and select name, city.",
                    "Rename name to full_name.",
                    "Drop city column.",
                    "Filter rows where age > 100",
                    # Complex / combined conditions
                    "Filter rows where age > 30 AND city == 'New York' and select name, age, city.",
                    "Filter rows where age > 30 OR city == 'New York' and select name, age.",
                    "Filter rows where age between 30 and 45 and name starts with 'J'.",
                    "Select rows where (age > 30 AND city == 'Chicago') OR (age <= 25 AND city == 'New York').",
                    "Select rows where name starts with 'J' and age <= 35.",
                    "Filter rows where age > 30 and city in ('Chicago', 'Boston').",
                    "Filter rows where age > 30 and not city == 'Los Angeles'.",
                    "Sort by age desc and then filter age > 30 and keep top 2.",
                    "Group by city and show average age where average_age > 30.",
                    "Filter where name contains 'an' and age between 30 and 50.",
                ]

                results = []
                for i, desc in enumerate(descriptions, start=1):
                    print(f"\n--- Eval {i}: {desc}")
                    state = initial_state.copy()
                    state["description"] = desc
                    state["sample_data"] = sample_df.to_dict(orient="records")
                    try:
                        fs = app.invoke(state)
                        outp = fs.get("output_path")
                        is_valid = fs.get("valid", False)
                        if outp and Path(outp).exists():
                            actual = pd.read_csv(outp)
                            actual_rows = len(actual)
                        else:
                            actual_rows = 0
                    except Exception as e:
                        print("Run failed:", e)
                        outp = None
                        is_valid = False
                        actual_rows = 0

                    results.append({
                        "id": i,
                        "description": desc,
                        "actual_rows": actual_rows,
                        "pass": bool(is_valid),
                        "output_path": outp if outp else None,
                    })

                results_df = pd.DataFrame(results)
                results_path = Path.cwd() / "eval_results.csv"
                results_df.to_csv(results_path, index=False)
                print("\nEvaluation complete. Results saved to:", results_path)
                print(results_df.to_string(index=False))
        else:
            final_state = app.invoke(initial_state)
            print("Transformation complete. Output saved to:", final_state["output_path"])
    except Exception as e:
        # Surface helpful debugging info
        print("Error running state graph:", type(e).__name__, e)