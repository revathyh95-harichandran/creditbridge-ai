# AI Usage Declaration: creditbridge-ai

**Author:** Revathy Harichandran
**AI tool used:** Claude Code (Anthropic)
**Last updated:** October 2026

## 1. Summary

I used Claude Code as an assistant during this project, mainly for writing code,
drafting documentation and figures, and routine tasks. The project idea, research
question, dataset selection, analysis plan, interpretation and all final decisions are
my own. I directed the work step by step through prompts I wrote myself, reviewed every
output, and asked for corrections whenever something was wrong or unclear. I understand
and can explain every part of the code, analysis and results in this project, and I take
full responsibility for its content.

## 2. Why I used an AI tool

I used the tool to make my work more productive and less time-consuming. Letting it
handle the routine coding meant I could spend more of my time on the parts that need
human judgement: brainstorming, asking analytical and logical questions, and exploring
the dataset more deeply and quickly than I could have alone. I treated the tool as an
assistant, not as the owner of the project.

## 3. What I did

- **Project idea and research question:** I came up with the idea of studying
  "thin-file" applicants (people with little credit history) and asked whether lenders
  turn them away for lack of information rather than real risk.
- **Literature review:** I used the tool to help search for and summarise research before
  starting. I read the key sources myself and checked every reference.
- **Dataset selection:** I chose the Home Credit Default Risk dataset from Kaggle and
  downloaded the files myself.
- **Data licensing:** I made sure the Kaggle data is never shared publicly, in line with
  the competition rules, and told the tool to keep it out of the GitHub repository.
- **Defining the segments:** I wrote the definition of "thin-file" (0–1 credit bureau
  records and at least one missing external credit score).
- **Planning every stage:** I decided the order of the project (set-up, data quality
  checks, cleaning, segment comparison, dashboard, model, model evaluation) and wrote
  specific instructions for each stage.
- **Cleaning rules:** I specified how missing values and outliers should be handled
  (median for numbers, keeping missing credit scores as a signal, an explicit "Missing"
  category, capping extreme values) and asked for every decision, and the alternatives
  that were ruled out, to be documented.
- **Analysis questions:** I chose what to compare between the two groups (default rate,
  income, employment length, age, application channel) and which non-bureau features to
  examine.
- **Power BI dashboard:** I designed and built the dashboard myself in Power BI Desktop.
  I asked for its numbers to be cross-checked against the data. When problems came up,
  I fixed them in Power BI, including finding the regional (locale) setting that was
  corrupting decimal numbers.
- **Model evaluation plan:** I designed the plan for testing the model: the accuracy
  trap, PR-AUC, Gini and KS, calibration, stability, a simple-rule comparison, results
  for different groups, and fairness. I also wrote the reasoning for which metric should
  be the main one.
- **Decisions:** whenever the tool offered options, I made the final call. Examples are
  in section 6.
- **Review and quality control:** I reviewed every output, compared results with my own
  earlier work, and asked for errors to be explained before going further.
- **Interpretation and conclusions:** I decided what the findings mean and what to
  recommend. The tool drafted the wording, which I reviewed and had rewritten in my own
  plain, simple style.
- **GitHub:** I created the repository, chose to make it public, and signed in to GitHub
  myself through Windows Credential Manager. The AI tool never had access to my password.

## 4. What the AI tool did

- **Wrote code** to my instructions: the Python scripts for loading, cleaning, segmenting,
  comparing, exporting, modelling, evaluating and charting, plus automated tests.
- **Ran the code** and reported the results back to me.
- **Suggested approaches** for some technical details, for example how to handle a
  placeholder value in the employment data, or a fairer way to compare models. I
  reviewed each suggestion before it was used.
- **Flagged problems** it found while running the code, which I then reviewed.
- **Drafted documentation:** README sections, reports and chart text, which I reviewed,
  edited and approved.
- **Created charts** from the data, to the requirements I set (plain-English titles,
  colour-blind-friendly colours, consistent colours).
- **Checked my dashboard's numbers** against the data when I asked.
- **Ran Git commands** (commits, pushes) when I asked. I did any sign-in myself.
- **Helped with research** for the literature review. I verified all sources.

## 5. How I stayed in control

- Every prompt was written by me. The tool did not act on its own goals.
- I worked in small, checked steps rather than asking for everything at once.
- For the write-ups and charts, I asked the tool to show them to me **before** they were
  written into files or published, and I approved them.
- Where my own earlier results differed from the tool's, I asked for the reason to be
  explained before continuing.
- Every number in the project can be reproduced by re-running the scripts, so nothing
  rests on the AI's word alone.
- Only the public, anonymised Kaggle dataset was used. No personal or confidential data
  was shared with the AI tool, and it never had access to my passwords.

## 6. Examples of decisions I made

- Keeping all Kaggle data, the database and the Power BI file out of the public repository.
- Using the *expected* result for "random order" rather than a single random shuffle.
- How to split applicants into 10 risk groups, matching the method I had used before.
- Putting all the findings in one README rather than splitting them into separate files.
- Writing every explanation in simple, plain English.
- Making the repository public, and keeping my personal email address private on GitHub.

## 7. What I learned

Through this project I learned why accuracy is misleading when only about 10% of people
default, how to check whether a model's percentages can be trusted (calibration), and how
a model can be unfair to a group even when that group's details aren't an input
(proxies).

## 8. Who did what

| Part of the project | Me | AI tool (Claude Code) |
|---|---|---|
| Project idea and research question | ✔ Did it | — |
| Literature review | ✔ Read and verified sources | Helped search and summarise |
| Dataset choice and download | ✔ Did it | — |
| Thin-file definition | ✔ Did it | Wrote the code |
| Data cleaning | ✔ Set the rules and approved decisions | Wrote the code, suggested options |
| Segment comparison | ✔ Chose what to compare | Wrote the code, ran it |
| Power BI dashboard | ✔ Designed, built and fixed it | Checked the numbers |
| Scoring model | ✔ Directed and approved it | Wrote and ran the code |
| Model evaluation | ✔ Designed the plan | Wrote the code and charts |
| Interpretation and recommendations | ✔ Decided them | Drafted the wording |
| Documentation | ✔ Reviewed and approved | Drafted it |
| GitHub | ✔ Created repo, signed in | Ran Git commands |
