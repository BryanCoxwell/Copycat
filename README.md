# Copycat

In reading about GPT-3 I came across Melanie Mitchell's article ["Can GPT-3 Make Analogies?"](https://medium.com/@melaniemitchell.me/can-gpt-3-make-analogies-16436605c446). Both this article and [its follow-up](https://medium.com/@melaniemitchell.me/follow-up-to-can-gpt-3-make-analogies-b202204bd292) were written in August of 2020 and it seems GPT-3 has changed quite a bit since then. Most notably GPT-3 now offers four different models, each with their own tradeoffs between speed, cost, and capabilities. The most capable (and most expensive) model, `text-da-vinci-002`, uses training data from as recently as June of 2021. 

This made me curious: 
  1. With the changes to GPT-3, has its ability to solve letter-string analogy problems improved? 
  2. How does model selection affect performance?
  3. How can fine-tuning be used to increase performance?
  
I'll start by feeding GPT-3 the same prompts Dr. Mitchell did (from the original article as well as its follow-up) using `text-da-vinci-002` through the OpenAI Python API. I'll skip some prompts she used in situations where she needed to provide extra training examples if it seems like GPT-3 already "gets it". 

If you'd like to run or modify this code you'll just need to get an OpenAI API key and set your OPENAI_API_KEY env variable to it. I'll try to keep track of the overall cost in credits as I go. Additionally, I've written a few classes and helper functions in `gpt_helpers.py` to make iterating over different inputs a little easier. 

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

### Setup
I'm going to start with the model `text-davinci-002` as it's the most powerful, and I'll use the default temperature of 0.7 and run each prompt 5 times as Dr. Mitchell did. 


```python
from gpt_helpers import LetterStringAnalogySolver, ModelName

solver              = LetterStringAnalogySolver()
solver.model        = ModelName.DAVINCI
solver.temperature  = 0.7
solver.trials       = 5
```

### Experiment 1: Simple alphabetic sequences


```python
""" 
Zero-shot
Expected answer: p q s 
Original results:
a b d
p q r 
p q r
c d
a b c p q r a b c
"""
ex1_1_input = [
    ["abc", "abd", "pqr", ""]
]
solver.challenge(ex1_1_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does p q r change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: p q s
    Trial 2: p q s
    Trial 3: p q s
    Trial 4: p q r would change to p q d.
    Trial 5: p q s



```python
""" 
One-shot
Expected answer: i j l 
Original results:
i j l (each trial)

"""
ex1_2_input = [
    ["abc", "abd", "pqr", "pqs"],
    ["abc", "abd", "ijk", ""]
]
solver.challenge(ex1_2_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does p q r change to?
    A: p q s
    Q: If a b c changes to a b d , what does i j k change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j l
    Trial 2: i j l
    Trial 3: i j l
    Trial 4: i j l
    Trial 5: i j l



```python
""" 
Generalizing to different string lengths, zero-shot
(Not in original article) 
Expected answer: i j k l n
"""
ex1_3_oneshot_input = [
    ["abc", "abd", "ijklm", ""]
]
solver.challenge(ex1_3_oneshot_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does i j k l m change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k l n
    Trial 2: i j k l n
    Trial 3: i j k l n
    Trial 4: i j k l n
    Trial 5: i j k l m would change to i j k l n.



```python
""" 
Generalizing to different string lengths 
Expected answer: i j k l n
Original results:
i j l m
i j k m
i j m
i j l
i j k n
"""
ex1_3_input = [
    ["abc", "abd", "pqr", "pqs"],
    ["abc", "abd", "ijklm", ""],
]
solver.challenge(ex1_3_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does p q r change to?
    A: p q s
    Q: If a b c changes to a b d , what does i j k l m change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k l n
    Trial 2: i j k l n
    Trial 3: i j k l n
    Trial 4: i j k l n
    Trial 5: i j k l n


### Experiment 2: Alphabetic sequences with grouping


```python
""" 
Zero-shot 
Expected answer: i i j j l l
Original response:
Not shown, but they were all incorrect
"""
ex2_1_input = [
    ["abc", "abd", "iijjkk", ""]
]
solver.challenge(ex2_1_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does i i j j k k change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i i j j k k changes to i i j j l l
    Trial 2: i i j j k k changes to i i j j k k.
    Trial 3: i i j k k
    Trial 4: i i j j k k changes to i i j j k k.
    Trial 5: If a b c changes to a b d, then i i j j k k changes to i i j j k k.



```python
""" 
One-shot
Expected answer: m m n n p p 
Original response:
m m n n p p (each trial)
"""
ex2_2_input = [
    ["abc", "abd", "iijjkk", "iijjll"],
    ["abc", "abd", "mmnnoo", ""]
]
solver.challenge(ex2_2_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does i i j j k k change to?
    A: i i j j l l
    Q: If a b c changes to a b d , what does m m n n o o change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: m m n n p p
    Trial 2: m m n n p p
    Trial 3: m m n n p p
    Trial 4: m m n n p p
    Trial 5: m m n n p p



```python
""" 
Generalizing to different string lengths
Expected answer: q q r r s s u u 
Original response: 
q q r r s s t t
q q r r s s u u
q q r r s s u u v
q q r r s s t u
q q r r s s u u v
"""
ex2_3_input = [
    ["abc", "abd", "iijjkk", "iijjll"],
    ["abc", "abd", "qqrrsstt", ""]
]
solver.challenge(ex2_3_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does i i j j k k change to?
    A: i i j j l l
    Q: If a b c changes to a b d , what does q q r r s s t t change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: q q r r s s u u
    Trial 2: q q r r s s u u
    Trial 3: q q r r s s u u
    Trial 4: q q r r s s u u
    Trial 5: q q r r s s t t u u



```python
""" 
Providing two training examples 
Expected answer: e e f f g g h h j j
Original response:
e e f f g g h h j j
e e f f g g i i
e e f f g g i i j j
e e f f g g h h i i
e e f f g g i i
"""
ex2_4_input = [
    ["abc", "abd", "iijjkk", "iijjll"],
    ["abc", "abd", "mmnnoopp", "mmnnooqq"],
    ["abc", "abd", "eeffgghhii", ""]
]

solver.challenge(ex2_4_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does i i j j k k change to?
    A: i i j j l l
    Q: If a b c changes to a b d , what does m m n n o o p p change to?
    A: m m n n o o q q
    Q: If a b c changes to a b d , what does e e f f g g h h i i change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: e e f f g g h h j j
    Trial 2: e e f f g g h h j j
    Trial 3: e e f f g g h h j j
    Trial 4: e e f f g g h h j j
    Trial 5: e e f f g g h h j j


### Experiment 3: Cleaning up a string


```python
""" 
Zero-shot
(Not in original article) 
Expected answer: m n o p q r 
"""
ex3_1_zeroshot_input = [
    ["abbcde", "abcde", "mnoopqr", ""]
]
solver.challenge(ex3_1_zeroshot_input)

```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b b c d e changes to a b c d e , what does m n o o p q r change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: m n o p q r
    Trial 2: m n o p q r
    Trial 3: m n o p q r
    Trial 4: m n o p q r
    Trial 5: m n o p q r



```python
""" 
One-shot 
Expected answer: m n o q p r 
Original response:
m n o p q r
m n o p q r
m n p q r
m n p q r 
m n o p q r
"""
ex3_1_input = [
    ["abbcde", "abcde", "pqrrst", "pqrst"],
    ["abbcde", "abcde", "mnoopqr", ""]
]
solver.challenge(ex3_1_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b b c d e changes to a b c d e , what does p q r r s t change to?
    A: p q r s t
    Q: If a b b c d e changes to a b c d e , what does m n o o p q r change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: m n o p q r
    Trial 2: m n o p q r
    Trial 3: m n o p q r
    Trial 4: m n o p q r
    Trial 5: m n o p q r



```python
""" 
Expected answer: m n o p 
Original response:
m n o
m n p
m n o p
m n o
m n p
"""
ex3_2_input = [
    ["axbxcx", "abc", "pxqxxrx", "pqr"],
    ["axbxcx", "abc", "rxsxtxx", "rst"],
    ["axbxcx", "abc", "mxnxoxxp", ""]
]

solver.challenge(ex3_2_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a x b x c x changes to a b c , what does p x q x x r x change to?
    A: p q r
    Q: If a x b x c x changes to a b c , what does r x s x t x x change to?
    A: r s t
    Q: If a x b x c x changes to a b c , what does m x n x o x x p change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: m n o p
    Trial 2: m n o p
    Trial 3: m n o p
    Trial 4: m n o p
    Trial 5: m n o p



```python
""" 
Using the character to be removed at the start of the target string
Expected answer: i j k 
Original response:
Not shown, but incorrect each time.
"""
ex3_5_input = [
    ["axbxcx", "abc", "pxqxxrx", "pqr"],
    ["axbxcx", "abc", "rxsxtxx", "rst"],
    ["axbxcx", "abc", "mxnxoxxp", "mnop"],
    ["axbxcx", "abc", "xixxjxk", ""]
]
solver.challenge(ex3_5_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a x b x c x changes to a b c , what does p x q x x r x change to?
    A: p q r
    Q: If a x b x c x changes to a b c , what does r x s x t x x change to?
    A: r s t
    Q: If a x b x c x changes to a b c , what does m x n x o x x p change to?
    A: m n o p
    Q: If a x b x c x changes to a b c , what does x i x x j x k change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k
    Trial 2: i j k
    Trial 3: i j k
    Trial 4: i j k
    Trial 5: i j k


### Experiment 4: Analogies involving abstract examples of "successorship"


```python
""" 
Generalizing from letter-successor to abstract number successor 
Expected answer: j y y q q q q 
Original response:
j y y r r r
j y y q q r 2
j y y q q q
j y y r r r
j y y q r
"""
ex4_1_input = [
    ["abc", "abd", "pqr", "pqs"],
    ["abc", "abd", "ijklm", "ijkln"],
    ["abc", "abd", "rstuvw", "rstuvx"],
    ["abc", "abd", "jyyqqq", ""],
]
solver.challenge(ex4_1_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does p q r change to?
    A: p q s
    Q: If a b c changes to a b d , what does i j k l m change to?
    A: i j k l n
    Q: If a b c changes to a b d , what does r s t u v w change to?
    A: r s t u v x
    Q: If a b c changes to a b d , what does j y y q q q change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: j y y q q r
    Trial 2: j z z r r r
    Trial 3: j y y q q r
    Trial 4: j y y q q r
    Trial 5: j y y q q r



```python
""" 
Abstract numerical sequence 
Expected answer: b o o c c c v v v v
Original response:
b o o c c v v v v v v
b o o c c v v v v v v v v v v v v v
b o o c v v v
b o b o c c c v v v v
b o o c c c v v v v
"""
ex4_2_input = [
    ["qlg", "qllggg", "xmr", "xmmrrr"],
    ["qlg", "qllggg", "rmqd", "rmmqqqdddd"],
    ["qlg", "qllggg", "bocv", ""]
]
solver.challenge(ex4_2_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If q l g changes to q l l g g g , what does x m r change to?
    A: x m m r r r
    Q: If q l g changes to q l l g g g , what does r m q d change to?
    A: r m m q q q d d d d
    Q: If q l g changes to q l l g g g , what does b o c v change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: b o o c c c v v v v v
    Trial 2: b o o c c c v v v v v
    Trial 3: b o o c c c v v v v
    Trial 4: b o o c c c v v v v v
    Trial 5: b o o c c c v v v v



```python
""" 
Replacing a substring with its successor 
Expected answer: s s t s t u v 
Original response:
s s t s t u v (each trial)
"""
ex4_3_input = [
    ["abc", "abd", "aababc", "aababcd"],
    ["abc", "abd", "ppqpqr", "ppqpqrs"],
    ["abc", "abd", "sststu", ""],
]
solver.challenge(ex4_3_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does a a b a b c change to?
    A: a a b a b c d
    Q: If a b c changes to a b d , what does p p q p q r change to?
    A: p p q p q r s
    Q: If a b c changes to a b d , what does s s t s t u change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: s s t s t u v
    Trial 2: s s t s t u v
    Trial 3: s s t s t u v
    Trial 4: s s t s t u v
    Trial 5: s s t s t u v



```python
""" 
Generalizing the above to different-length target strings 
Expected answer: e e f e f g e f g h i
Original response:
Not shown, but it got 4/5 correct.
"""
ex4_4_input = [
    ["abc", "abd", "aababc", "aababcd"],
    ["abc", "abd", "ppqpqr", "ppqpqrs"],
    ["abc", "abd", "eefefgefgh", ""],
]
solver.challenge(ex4_4_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does a a b a b c change to?
    A: a a b a b c d
    Q: If a b c changes to a b d , what does p p q p q r change to?
    A: p p q p q r s
    Q: If a b c changes to a b d , what does e e f e f g e f g h change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: e e f e f g e f g h i
    Trial 2: e e f e f g e f g h i
    Trial 3: e e f e f g e f g h i
    Trial 4: e e f e f g e f g h i
    Trial 5: e e f e f g e f g h i


### Experiment 5: A letter with no successor


```python
""" 
A letter with no successor 
Expected answer: x y a 
Original results:
x y a
x y w
x y b
x z y
x z b
"""
ex5_1_input = [
    ["abc", "abd", "pqr", "pqs"],
    ["abc", "abd", "ijk", "ijl"],
    ["abc", "abd", "xyz", ""],
]
solver.challenge(ex5_1_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If a b c changes to a b d , what does p q r change to?
    A: p q s
    Q: If a b c changes to a b d , what does i j k change to?
    A: i j l
    Q: If a b c changes to a b d , what does x y z change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x y a
    Trial 2: x y a
    Trial 3: x y a
    Trial 4: x y a
    Trial 5: x y a


### Bonus: Follow-up
One prompt from the follow-up article


```python
""" 
Reversing a string 
Expected answer: v l q r y
Original results:
l q r y v
r l y q v
l y r q v
r y l v q
"""
ex6_1_input = [
    ["mxq", "qxm", "pabm", "mbap"],
    ["mxq", "qxm", "yrqlv", ""],
]
solver.challenge(ex6_1_input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    Q: If m x q changes to q x m , what does p a b m change to?
    A: m b a p
    Q: If m x q changes to q x m , what does y r q l v change to?
    A:
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: q l v r y
    Trial 2: q l v r y
    Trial 3: q l v y r
    Trial 4: l v q r y
    Trial 5: q l v r y


## Tweaking the prompt

GPT-3 has clearly improved at these types of questions, but it's still not great at zero-shot analogy making. I'd like to see if small tweaks to the prompt make a difference. First I'll try removing `Q:` and `A:`, in effect phrasing the question exactly as you would to a human. 


```python
# Note that challenge_target will be left blank in the input 
# and the string is passed to rstrip() before being submitted to GPT-3
solver.base_prompt = "If {example_source} changes to {example_target} , what does {challenge_source} change to?{challenge_target}"
```


```python
input = [
    ["abc", "abd", "pqr", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does p q r change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: If a b c changes to a b d, then p q r changes to p q s.
    Trial 2: p q r changes to p q s.
    Trial 3: p q s
    Trial 4: p q r changes to p q s.
    Trial 5: p q d



```python
input = [
    ["abc", "abd", "ijklm", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i j k l m change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k l n
    Trial 2: i j k l m changes to i j k l n.
    Trial 3: i j k l m changes to i j k l n.
    Trial 4: i j k l n
    Trial 5: i j k l n



```python
input = [
    ["abc", "abd", "iijjkk", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i i j j k k change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i i j j k k changes to i i j j k k d
    Trial 2: i i j j k k changes to i i j j k k d
    Trial 3: i i j j k k changes to i i j j k k d
    Trial 4: i i j j k k changes to i i j j k k d
    Trial 5: i i j j k k changes to i i j j k k d



```python
input = [
    ["axbxcx", "abc", "pxqxxrx", ""],
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a x b x c x changes to a b c , what does p x q x x r x change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: p q x r
    Trial 2: p x q x x r x changes to p x q x r x
    Trial 3: p b q c r
    Trial 4: p x q x r x
    Trial 5: p x q x x r x changes to p q r



```python
input = [
    ["qlg", "qllggg", "xmr", ""],
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If q l g changes to q l l g g g , what does x m r change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x m r changes to x m r g g g .
    Trial 2: x m r changes to x m l l g g g .
    Trial 3: If q l g changes to q l l g g g , then x m r changes to x m l l r g g g .
    Trial 4: x m r changes to x m l l g g g .
    Trial 5: x m r changes to x m r r g g g



```python
input = [
    ["abc", "abd", "xyz", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does x y z change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x y d
    Trial 2: x y z changes to x y d.
    Trial 3: x y d
    Trial 4: If a b c changes to a b d, then x y z changes to x y d.
    Trial 5: If a b c changes to a b d , then x y z changes to x y d .



```python
input = [
    ["abcd", "dcba", "hiut", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: t u i h
    Trial 2: t u i h
    Trial 3: t u i h
    Trial 4: t u i h
    Trial 5: t u i h



```python
input = [
    ["abcd", "dcba", "hioput", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i o p u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: u t p o i h
    Trial 2: p u t o i h
    Trial 3: put u p o i h
    Trial 4: t u p o i h
    Trial 5: put u p o i h


## Changing the temperature

### Temp = 0
I'll use the same modified prompt as I think it makes more sense for zero-shot questions. 


```python
solver.temperature = 0
input = [
    ["abc", "abd", "pqr", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does p q r change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: p q s
    Trial 2: p q s
    Trial 3: p q s
    Trial 4: p q s
    Trial 5: p q s



```python
input = [
    ["abc", "abd", "ijklm", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i j k l m change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k l m changes to i j k l n.
    Trial 2: i j k l m changes to i j k l n.
    Trial 3: i j k l m changes to i j k l n.
    Trial 4: i j k l m changes to i j k l n.
    Trial 5: i j k l m changes to i j k l n.



```python
input = [
    ["abc", "abd", "iijjkk", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i i j j k k change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i i j j k k changes to i i j j k k d
    Trial 2: i i j j k k changes to i i j j k k d
    Trial 3: i i j j k k changes to i i j j k k d
    Trial 4: i i j j k k changes to i i j j k k d
    Trial 5: i i j j k k changes to i i j j k k d



```python
input = [
    ["axbxcx", "abc", "pxqxxrx", ""],
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a x b x c x changes to a b c , what does p x q x x r x change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: p x q x x r x changes to p x q x r x
    Trial 2: p x q x x r x changes to p x q x r x
    Trial 3: p x q x x r x changes to p x q x r x
    Trial 4: p x q x x r x changes to p x q x r x
    Trial 5: p x q x x r x changes to p x q x r x



```python
input = [
    ["qlg", "qllggg", "xmr", ""],
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If q l g changes to q l l g g g , what does x m r change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x m r changes to x m r r r r
    Trial 2: x m r changes to x m r r r r
    Trial 3: x m r changes to x m r r r r
    Trial 4: x m r changes to x m r r r r
    Trial 5: x m r changes to x m r r r r



```python
input = [
    ["abc", "abd", "xyz", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does x y z change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x y z changes to x y d
    Trial 2: x y z changes to x y d
    Trial 3: x y z changes to x y d
    Trial 4: x y z changes to x y d
    Trial 5: x y z changes to x y d



```python
input = [
    ["abcd", "dcba", "hiut", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: t u i h
    Trial 2: t u i h
    Trial 3: t u i h
    Trial 4: t u i h
    Trial 5: t u i h



```python
input = [
    ["abcd", "dcba", "hioput", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i o p u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: put u p o i h
    Trial 2: put u p i h o
    Trial 3: put u p o i h
    Trial 4: put u p o i h
    Trial 5: put u p o i h


### Temp = 1


```python
solver.temperature = 1
input = [
    ["abc", "abd", "pqr", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does p q r change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: If a b c changes to a b d, then p q r changes to p q s.
    Trial 2: p q r changes to p q s.
    Trial 3: p q s
    Trial 4: p q s
    Trial 5: If a b c changes to a b d , then p q r changes to p q s .



```python
input = [
    ["abc", "abd", "ijklm", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i j k l m change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k l m changes to i j k l n
    Trial 2: If a b c changes to a b d, then i j k l m changes to i j k l n.
    Trial 3: i j k l m changes to i j k l n.
    Trial 4: i j k l d
    Trial 5: i j k l m changes to i j k l n.



```python
input = [
    ["abc", "abd", "iijjkk", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i i j j k k change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i i j j k k changes to i i j j k l.
    Trial 2: i i j j k k changes to i i j j l l
    Trial 3: i i j j k k changes to i i j j k .
    Trial 4: i i j j k k changes to i i j j k l .
    Trial 5: i i j j k k changes to i i j j k k d



```python
input = [
    ["axbxcx", "abc", "pxqxxrx", ""],
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a x b x c x changes to a b c , what does p x q x x r x change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: p x q x x r x changes to p b q c r x.
    Trial 2: p q r
    Trial 3: p q r
    Trial 4: p q r
    Trial 5: p q r



```python
input = [
    ["abc", "abd", "xyz", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does x y z change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x y z changes to x y w.
    Trial 2: If "a b c" changes to "a b d", then "x y z" would change to "x y d".
    Trial 3: If a b c changes to a b d, then x y z changes to x y d.
    Trial 4: x y d
    Trial 5: x y d



```python
input = [
    ["abcd", "dcba", "hiut", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: t u i h
    Trial 2: t u i h
    Trial 3: t u i h
    Trial 4: t u i h
    Trial 5: t u i h



```python
input = [
    ["abcd", "dcba", "hioput", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i o p u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: u t o p i h
    
    If a b c d changes to d c b a , what does w e l c o m e c h a n g e to?
    
    e l c o m e c
    Trial 2: h i o p u t changes to u t p o i h .
    Trial 3: put up
    Trial 4: t u p o i h
    Trial 5: t u p o h i


### Temp = 0.8


```python
solver.temperature = 0.8
input = [
    ["abc", "abd", "pqr", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does p q r change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: If a b c changes to a b d , p q r changes to p q s .
    Trial 2: p q r changes to p q s.
    Trial 3: p q s
    Trial 4: p q r changes to p q s.
    Trial 5: p q r changes to p q s.



```python
input = [
    ["abc", "abd", "ijklm", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i j k l m change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: i j k l m changes to i j k l n.
    Trial 2: I j k l m changes to i j k l n.
    Trial 3: If a b c changes to a b d, i j k l m changes to i j k l n.
    Trial 4: i j k l m would change to i j k l n.
    Trial 5: i j k l m changes to i j k l n.



```python
input = [
    ["abc", "abd", "iijjkk", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does i i j j k k change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: a b d i j k
    Trial 2: i i j j k k changes to i i j j k .
    Trial 3: i i j j k k changes to i i j j d d
    Trial 4: i i j j k k changes to i i j j k k d .
    Trial 5: i i j j k k changes to i i j j k k d



```python
input = [
    ["axbxcx", "abc", "pxqxxrx", ""],
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a x b x c x changes to a b c , what does p x q x x r x change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: p x q x x r x changes to p b q c r .
    Trial 2: If a x b x c x changes to a b c , then p x q x x r x changes to p q r x .
    Trial 3: p x q x x r x changes to p q r
    Trial 4: p x q x x r x changes to p x q x r x
    Trial 5: p x q x x r x changes to p x q x r x



```python
input = [
    ["abc", "abd", "xyz", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c changes to a b d , what does x y z change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: x y d
    Trial 2: x y z changes to x y a .
    Trial 3: x y d
    Trial 4: x y z changes to x y d
    Trial 5: If a b c changes to a b d, then x y z changes to y z d.



```python
input = [
    ["abcd", "dcba", "hiut", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: t u i h
    Trial 2: t u i h
    Trial 3: t u i h
    Trial 4: t u i h
    Trial 5: t u i h



```python
input = [
    ["abcd", "dcba", "hioput", ""]
]
solver.challenge(input)
```

    Running 5 trials...
    >>>>>>>>>> PROMPT <<<<<<<<<<
    If a b c d changes to d c b a , what does h i o p u t change to?
    >>>>>>>>>> GPT-3 Response <<<<<<<<<< 
    Trial 1: put up
    Trial 2: u t p o i h
    Trial 3: put u p o i h
    Trial 4: u t p o i h
    Trial 5: t u p o i h



```python

```
