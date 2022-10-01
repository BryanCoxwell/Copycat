# Copycat

In reading about GPT-3 I came across Melanie Mitchell's article ["Can GPT-3 Make Analogies?"](https://medium.com/@melaniemitchell.me/can-gpt-3-make-analogies-16436605c446). Both this article and [its follow-up](https://medium.com/@melaniemitchell.me/follow-up-to-can-gpt-3-make-analogies-b202204bd292) were written in August of 2020 and it seems GPT-3 has changed quite a bit since then. Most notably GPT-3 now offers four different models, each with their own tradeoffs between speed, cost, and capabilities. The most capable (and most expensive) model, `text-da-vinci-002`, uses training data from as recently as June of 2021. 

This made me curious: 
  1. With the changes to GPT-3, has its ability to solve letter-string analogy problems improved? 
  2. How does model selection affect performance?
  3. How can fine-tuning be used to increase performance?
  
I'll start by feeding GPT-3 the same prompts Dr. Mitchell did (from the original article as well as its follow-up) using `text-da-vinci-002` through the OpenAI Python API. I'll skip some prompts she used in situations where she needed to provide extra training examples if it seems like GPT-3 already "gets it". 

If you'd like to run or modify this code you'll just need to get an OpenAI API key and set your OPENAI_API_KEY env variable to it. I'll try to keep track of the overall cost in credits as I go. Additionally, I've written a few classes and helper functions in `gpt_helpers.py` to make iterating over different inputs a little easier. 

## Project Structure
`copycat.ipynb` contains the original prompts from the above article above, some variations of those prompts, and various parameter changes. 
`copycat_r2.ipynb` contains prompts using the same patterns as the original prompts, but with new characters (Dr. Mitchell pointed out that her prompts were likely used in the model's training data.)
`gpt_helpers.py` includes the code used to run the prompts and is explained further below. 

## How gpt_helpers.py works
The `LetterStringAnalogySolver` class configures the parameters passed to GPT-3, formats input, and displays the response. Configurable GPT-3 parameters are limited to the model name (required) and temperature for now (`max_tokens` is set as a constant).
The base prompt which the inputs are formatted into is also configurable. If not set it will default to:

```
"Q: if {example_source} changes to {example_target} , what does {challenge_source} change to?\nA: {challenge_target}"
```
If modified, the only requirement on the base prompt is that it includes the same format variables `example_source`, `example_target`, `challenge_source`, and `challenge_target`.

The input (prompt data) is a list of lists of strings which will be formatted (in order!) into the base prompt. Formatting includes inserting a space between each character (to avoid issues caused by GPT-3's byte-pair encoding), and cases are preserved. So, for example:
```
input = [
    ["aaa", "bbb", "ccc", "ddd"],
    ["fff", "ggg", "hhh", ""]
  ]
```
would yield the prompt
```
Q: if a a a changes to b b b , what does c c c change to?
A: d d d
Q: if f f f changes to g g g , what does h h h change to?
A:
```
Note that the last element of the last list is empty since we want GPT-3 to tell us what it thinks the `challenge_target` is. 

To pass the input to GPT-3 and receive a response, pass the prompt data to `LetterStringAnalogySolver.challenge()`.
To run each request multiple times, set the `trials` parameter. 