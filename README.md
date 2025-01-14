# RD Table Bench Invocation + Grading Code

This repo contains the code for invoking each provider and grading the results.

## Installing Dependencies

```
pip install -r requirements.txt
```

## Downloading Data

https://huggingface.co/datasets/reducto/rd-tablebench/blob/main/README.md

## Env Vars

Create an `.env` file with the following:

```
INPUT_DIR=
OUTPUT_DIR=

# note: only need keys for providers you want to use
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
...
```

## Parsing

`python providers/llm.py --model gemini-2.0-flash-exp --num-workers 10`

## Grading

`python grade_cli.py --model gemini-2.0-flash-exp`
