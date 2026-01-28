import time

import click

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments,Trainer


PROMPT_TEMPLATE = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{question} [/INST]
{response}
"""

def ask_question(model, tokenizer, question):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    prompt = PROMPT_TEMPLATE.format(question=question, response="")
    tokens_in = len(tokenizer(prompt)["input_ids"])
    start = time.time()
    result = pipe(prompt)
    end = time.time()
    generated_text = result[0]['generated_text']
    tokens_out = len(tokenizer(generated_text)["input_ids"])
    print(generated_text)
    tokens_generated = tokens_out - tokens_in
    time_taken = end - start
    tokens_per_second = tokens_generated / time_taken
    print(f"{tokens_generated} tokens in {time_taken:.2f}s: {tokens_per_second:.2f} tokens/s)")


def tokenize_function(tokenizer, examples):
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized


@click.command()
@click.argument("base_model")
def main(base_model):
    dataset_source = "gpjt/openassistant-guanaco-llama2-format"

    dataset = load_dataset(dataset_source)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenized_dataset = dataset.map(lambda examples: tokenize_function(tokenizer, examples), batched=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda", trust_remote_code=True)

    print("Before training sample:")
    ask_question(model, tokenizer, "Who is Leonardo Da Vinci?")

    batch_size = 6
    args = TrainingArguments(
        'outputs', 
        learning_rate=8e-5, 
        warmup_ratio=0.1, 
        lr_scheduler_type='cosine', 
        fp16=True,
        eval_strategy="epoch", 
        eval_on_start=True,
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=1, 
        weight_decay=0.01, 
        report_to='none'
    )

    trainer = Trainer(
        model, args, 
        train_dataset=tokenized_dataset['train'], 
        eval_dataset=tokenized_dataset['test'],
        processing_class=tokenizer,
    )

    trainer.train()

    print("After training sample:")
    ask_question(model, tokenizer, "Who is Leonardo Da Vinci?")


if __name__ == "__main__":
    main()
