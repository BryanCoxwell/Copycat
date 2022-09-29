import os
import openai
from enum import StrEnum
from typing import Optional

openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_BASE_PROMPT = "Q: If {example_source} changes to {example_target} , what does {challenge_source} change to?\nA: {challenge_target}"
MAX_TOKENS          = 48
MAX_TRIALS          = 20

class ModelName(StrEnum):
    DAVINCI = "text-davinci-002"
    CURIE   = "text-curie-001"
    BABBAGE = "text-babbage-001"
    ADA     = "text-ada-001"


class LetterStringAnalogySolver:
    def __init__(
            self,
            model: Optional[ModelName]      = None,
            temperature: Optional[float]    = None,
            base_prompt: str                = DEFAULT_BASE_PROMPT,
            trials: int                     = 1):
        self._model             = model
        self._base_prompt       = base_prompt
        self._temperature       = temperature
        self._trials            = trials

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_name: ModelName | str):
        if type(model_name) == ModelName:
            self._model = model_name
        elif type(model_name) == str:
            self._model = ModelName(model_name)
        else:
            raise TypeError("Model name must be of type str or ModelName")

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, val: float):
        if not 0 <= val <= 1:
            raise ValueError("Temperature must be a value between 0 and 1")
        self._temperature = val

    @property
    def base_prompt(self):
        return self._base_prompt

    @base_prompt.setter
    def base_prompt(self, prompt: str):
        format_vars = ["{example_source}", "{example_target}", "{challenge_source}"]
        for format_var in format_vars:
            if not format_var in prompt:
                raise ValueError(f"Base prompt missing format variable: {format_var}")
        self._base_prompt = prompt

    @property
    def trials(self):
        return self._trials

    @trials.setter
    def trials(self, val):
        if val > MAX_TRIALS:
            raise ValueError(f"Can't run more than {MAX_TRIALS} trials.")
        self._trials = val

    def challenge(self, prompt_data: list):
        if not self._model:
            print(""" 
            No model selected! Choose a model and try again.
            The model can be configured by using its setter which accepts either a ModelName or a string:
            <my_solver_object>.model = ModelName.DAVINCI
            or 
            <my_solver_object>.model = 'text-davinci-002'
            """)
            return
        print(f"Running {self._trials} trials...")
        prompt = self.generate_prompt(prompt_data)
        print(">>>>>>>>>> PROMPT <<<<<<<<<<")
        print(prompt)
        print(">>>>>>>>>> GPT-3 Response <<<<<<<<<< ")
        for trial in range(self._trials):
            self.display_completion(self.get_completion(prompt), trial)

    def get_completion(self, prompt: str):
        response = openai.Completion.create(
            model               = self._model,
            prompt              = prompt,
            temperature         = self._temperature,
            max_tokens          = MAX_TOKENS,
        )
        return response

    def display_completion(self, completion, trial):
        print(f"Trial {trial+1}: {completion['choices'][0]['text'].strip()}")

    def generate_prompt(self, input_data) -> str:
        prompt = ""
        for letter_string_list in input_data:
            formatted_input_data = [" ".join(letter_string.replace(" ", "")) for letter_string in letter_string_list]
            prompt += self._base_prompt.format(
                example_source      = formatted_input_data[0],
                example_target      = formatted_input_data[1],
                challenge_source    = formatted_input_data[2],
                challenge_target    = formatted_input_data[3],
            )
            if formatted_input_data[3]:
                prompt += "\n"
        return prompt[0:-3].rstrip()


if __name__ == "__main__":
    solver              = LetterStringAnalogySolver()
    solver.temperature  = 1
    solver.model        = ModelName.DAVINCI
    input = [
        ["aaa", "bbb", "ccc", "ddd"],
        ["fff", "ggg", "hhh", ""]
    ]
    print(solver.generate_prompt(input_data=input))



