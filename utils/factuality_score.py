import json
from openai import OpenAI

client = OpenAI()

BREAKDOWN_PROMPT = (
    "Please breakdown the following input into a set of small, independent claims, "
    "and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n"
    "The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts "
    "and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult "
    "for anyone to know, like the birthdays of non-notable people. The input is: "
)

BREAKDOWN_SYSTEM_PROMPT = (
    "You are a helpful assistant whose job is to break down your inputs "
    "into a set of small claims so that a human can easily check each one. "
    "Make sure that each claim is small and non-overlapping."
)


def get_subclaims(output: str, breakdown_prompt: str, factuality_model: str):
    """Break down the output into subclaims using GPT-4-mini."""
    prompt = breakdown_prompt + output
    response = client.chat.completions.create(
        model=factuality_model,
        messages=[
            {"role": "system", "content": BREAKDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    response = response.choices[0].message.content

    # Parse the JSONL response
    subclaims = []
    for line in response.strip().split("\n"):
        try:
            claim_data = json.loads(line)
            subclaims.append(claim_data)
        except json.JSONDecodeError:
            continue
    return subclaims


def evaluate_factuality_subclaims(
    subclaims: list[dict], question: str, answer: str, factuality_model: str
):
    """Evaluate the factuality of subclaims using web search."""
    factual_claims = 0
    total_claims = len(subclaims)

    # Prepare all claims for batch processing
    claims_list = [claim["subclaim"] for claim in subclaims]
    claims_text = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims_list)])

    # Use web search to verify all claims at once
    prompt = f"""Given the question: {question}, and answer to the question: {answer},
    please verify if each of these claims is factual.
    Claims:
    {claims_text}
    
    Return your answer as a JSON array, where each element is an object with these keys:
    {{"subclaim": "[CLAIM]", "factual": 1 or 0, "reason": "explanation with reference to search results", "source": "source url"}}
    
    Format your response as a valid JSON array only, with no additional text or formatting.
    Example format:
    [
        {{"subclaim": "claim 1", "factual": 1, "reason": "explanation", "source": "source"}},
        {{"subclaim": "claim 2", "factual": 0, "reason": "explanation", "source": "source"}}
    ]"""

    response = client.responses.create(
        model=factuality_model,
        tools=[{"type": "web_search_preview", "search_context_size": "low"}],
        input=prompt,
    )
    response_content = response.output_text

    # Clean up the response content
    response_content = response_content.replace("```jsonl\n", "")
    response_content = response_content.replace("\\", "\\\\")
    response_content = response_content.replace("```", "")
    response_content = response_content.replace("json", "")
    response_content = response_content.strip()

    annotation = []
    try:
        # Try parsing as a JSON array first
        try:
            results = json.loads(response_content)
            if isinstance(results, list):
                for result in results:
                    if type(result["factual"]) == str:
                        result["factual"] = int(result["factual"])
                    assert result["factual"] in [
                        0,
                        1,
                    ], f"factual should be 0 or 1, but got {result['factual']}"
                    factual_claims += result["factual"]
                    annotation.append(
                        {
                            "subclaim": result["subclaim"],
                            "factual": result["factual"],
                            "reason": result.get("reason", ""),
                            "source": result.get("source", ""),
                        }
                    )
            else:
                raise ValueError("Response is not a JSON array")
        except json.JSONDecodeError:
            # If JSON array parsing fails, try parsing line by line
            for line in response_content.splitlines():
                if not line.strip():
                    continue
                # Clean up the line
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    if type(data["factual"]) == str:
                        data["factual"] = int(data["factual"])
                    assert data["factual"] in [
                        0,
                        1,
                    ], f"factual should be 0 or 1, but got {data['factual']}"
                    factual_claims += data["factual"]
                    annotation.append(
                        {
                            "subclaim": data["subclaim"],
                            "factual": data["factual"],
                            "reason": data.get("reason", ""),
                            "source": data.get("source", ""),
                        }
                    )
    except Exception as e:
        print(f"Error processing response: {e}")
        print("Response content:", response_content)
        # Create default responses for all claims
        for claim in claims_list:
            annotation.append(
                {"subclaim": claim, "factual": 0, "reason": "parsing_error", "source": ""}
            )

    # Calculate the factuality score
    factuality_score = factual_claims / total_claims if total_claims > 0 else 0.0
    annotation = json.dumps(annotation, indent=4, ensure_ascii=False)
    return factuality_score, annotation


def evaluate_factuality(
    question: str, answer: str, factuality_model: str, breakdown_prompt: str = None
):
    """Evaluate the factuality of the answer using web search."""
    if breakdown_prompt is None:
        breakdown_prompt = BREAKDOWN_PROMPT
    subclaims = get_subclaims(answer, breakdown_prompt, factuality_model)
    factuality_score, annotation = evaluate_factuality_subclaims(
        subclaims, question, answer, factuality_model
    )
    return factuality_score, annotation
