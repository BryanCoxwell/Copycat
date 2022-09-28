import os
import openai
from enum import StrEnum
from typing import Optional

openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_BASE_PROMPT = "if {example_source} changes to {example_target} , what does {challenge_source} change to?"

class ModelName(StrEnum):
    DAVINCI = "text-davinci-002"
    CURIE   = "text-curie-001"
    BABBAGE = "text-babbage-001"
    ADA     = "text-ada-001"

class LetterStringAnalogySolver:
	def __init__(self, 
				model: Optional[ModelName] 		= None, 
				temperature: Optional[float] 	= None,
				base_prompt: str 				= DEFAULT_BASE_PROMPT,
				max_tokens: int 				= 64,
				top_p: int 						= 1,
				frequency_penalty: int 			= 0,
				presence_penalty: int 			= 0):
		self._model		 		= model
		self._base_prompt 		= base_prompt
		self._temperature 		= temperature
		self._max_tokens 		= max_tokens
		self._top_p 			= top_p
		self._frequency_penalty = frequency_penalty
		self._presence_penalty 	= presence_penalty

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

	def challenge(self, example_source: str, example_target: str, challenge_source: str):
		prompt = self.generate_prompt(example_source, example_target, challenge_source)
		self.get_completion(prompt)

	def get_completion(self, prompt: str):
		response = openai.Completion.create(
			model				= self._model,
			prompt				= prompt,
			temperature			= self._temperature,
			max_tokens			= self._max_tokens,
			top_p				= self._top_p,
			frequency_penalty	= self._frequency_penalty,
			presence_penalty	= self._presence_penalty
		)
		return response

	def generate_prompt(self, example_source: str, example_target: str, challenge_source: str) -> str:
		return self._base_prompt.format(
			example_source 		= self.format_letter_string(example_source),
			example_target 		= self.format_letter_string(example_target),
			challenge_source 	= self.format_letter_string(challenge_source)
		)

	def format_letter_string(self, letter_string: str) -> str:
		""" Ensures a space between each character, but doesn't add one if they're already there. """
		return " ".join(letter_string.replace(" ", ""))


if __name__=="__main__":
	solver = LetterStringAnalogySolver()
	solver.model = "text-ada-001"
	solver.temperature = 0.1
	print(solver.generate_prompt("asd", "dfds", "dfs"))