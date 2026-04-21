import textgrad as tg
from textgrad.variable import Variable


def prepare_multi_choice_example(example):
    choices = "\n".join(f"{chr(65 + i)}. {ans}" for i, ans in enumerate(example["answers"]))
    question_text = f"{example['question']}\n\n{choices}"
    correct_answer = chr(65 + example["label"])

    x = tg.Variable(
        question_text,
        requires_grad=False,
        role_description="multiple choice question",
    )
    y_true = tg.Variable(
        correct_answer,
        requires_grad=False,
        role_description="correct answer for the multiple choice question",
    )
    return x, y_true


def prepare_integer_math_example(example):
    question_text = f"{example['question']}"
    if "####" in str(example["answer"]):
        correct_answer = str(int(str(example["answer"]).split("####")[1].strip().replace(",", "")))
    else:
        correct_answer = str(int(str(example["answer"])))

    x = Variable(
        question_text,
        requires_grad=False,
        role_description="math question",
    )
    y_true = Variable(
        correct_answer,
        requires_grad=False,
        role_description="correct answer for the math question",
    )
    return x, y_true


def prepare_factscore_example(example):
    question_text = f"{example['question']}"
    x = Variable(
        question_text,
        requires_grad=False,
        role_description="factscore question",
    )

    y_true = Variable(
        "",
        requires_grad=False,
        role_description="correct answer for the factscore question",
    )
    return x, y_true


def prepare_qasper_example(example):
    question_text = f"{example['question']}"
    context_text = f"{example['context']}"
    formatted_question = f"Context: {context_text}\n\n Question: {question_text}\n\n"
    question_type = example["question_type"]
    if question_type != "freeform":
        print(f"Noticed question type {question_type}")

    unanswerable_flag = example["unanswerable"]
    if unanswerable_flag:
        correct_answer = "Unanswerable"
    else:
        correct_answer = example["answer"]

    x = Variable(
        formatted_question,
        requires_grad=False,
        role_description="open-ended question with context/evidence",
    )
    y_true = Variable(
        correct_answer,
        requires_grad=False,
        role_description="correct answer for the open-ended question",
    )
    return x, y_true


def prepare_simpleqa_example(example):
    x = Variable(
        example["question"],
        requires_grad=False,
        role_description="open-ended question",
    )
    y_true = Variable(
        example["answer"],
        requires_grad=False,
        role_description="correct answer for the open-ended question",
    )
    return x, y_true
