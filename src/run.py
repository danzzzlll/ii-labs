import warnings
import logging
from transformers import pipeline
import re
# import time

warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)


def generate_response(user_input, generation_model):
    modify_input = f'Вопрос: {user_input} Ответ:'
    response = generation_model(modify_input, **config)[0]['generated_text']
    response = re.sub(f'{modify_input}', '', response)
    return response


def main():
    print('Loading model ... ')
    model_name = 'danzzzll/verystupid_rugpt-3'
    generation_model = pipeline('text-generation', model=model_name, tokenizer=model_name, device='cpu')
    print('Loading end.')

    config = {
        "max_length": 100,
        "min_length": 30,
        "temperature": .9,
        "num_beams": 3,
        "repetition_penalty": 1.5,
        "num_return_sequences": 2,
        "no_repeat_ngram_size": 2,
        "do_sample": True
    }

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            break

        response = generate_response(user_input, generation_model, config)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
