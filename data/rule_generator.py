import random

def generate_implication_chain(variables):
    rules = [f"{variables[i]} -> {variables[i+1]}" for i in range(len(variables) - 1)]
    premise = variables[0]
    conclusion = variables[-1]
    input_text = "; ".join(rules + [premise])
    target_text = conclusion
    return input_text, target_text


def generate_boolean_task():
    operators = ["AND", "OR"]
    op = random.choice(operators)
    a, b = random.randint(0, 1), random.randint(0, 1)

    if random.random() < 0.3:
        val = 1 - (a & b if op == "AND" else a | b)
        input_text = f"NOT {a} {op} {b}"
        target_text = str(val)
    else:
        val = a & b if op == "AND" else a | b
        input_text = f"{a} {op} {b}"
        target_text = str(val)

    return input_text, target_text


def generate_multistep_boolean_task():
    operators = ["AND", "OR"]
    num_ops = random.randint(2, 3)
    values = [random.randint(0, 1) for _ in range(num_ops + 1)]
    expr_parts = []

    for i in range(num_ops):
        val = values[i]
        val_str = f"NOT {val}" if random.random() < 0.3 else str(val)
        expr_parts.append(val_str)
        expr_parts.append(operators[i % 2])

    last_val = values[-1]
    expr_parts.append(f"NOT {last_val}" if random.random() < 0.3 else str(last_val))
    expr = " ".join(expr_parts)

    eval_expr = expr.replace("AND", "&").replace("OR", "|").replace("NOT", "1 -")
    try:
        result = eval(eval_expr)
    except Exception:
        result = 0

    return expr, str(result)


def generate_unique_dataset(num_samples=100):
    dataset = []
    seen = set()  # keep track of unique input-target pairs

    while len(dataset) < num_samples:
        task_type = random.choices(
            ["implication", "boolean", "multistep_boolean"],
            weights=[0.25, 0.25, 0.3, 0.2]
        )[0]

        if task_type == "implication":
            symbols = ["A", "B", "C", "D", "E", "F"]
            length = random.randint(2, 4)
            vars_sample = random.sample(symbols, length)
            pair = generate_implication_chain(vars_sample)
        elif task_type == "boolean":
            pair = generate_boolean_task()
        elif task_type == "multistep_boolean":
            pair = generate_multistep_boolean_task()
       
        # Only add if not already seen
        if pair not in seen:
            seen.add(pair)
            dataset.append(pair)

    return dataset


def write_txt(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for inp, tgt in dataset:
            f.write(f"{inp}\t{tgt}\n")


if __name__ == "__main__":
    data = generate_unique_dataset(num_samples=850)
    write_txt("logic-solution.txt", data)
    print("Wrote logic-solution.txt (all unique)")