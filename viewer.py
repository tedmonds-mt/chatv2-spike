import gradio as gr

from invoke import invoke_agent

demo = gr.ChatInterface(
    fn=invoke_agent,
    title="Prototype",
    description="I'm a lightweight version of Gov UK chat. Ask me stuff.",
)

if __name__ == "__main__":
    demo.launch()

"""
user: Uk retirement age?

assistant: <classification_analysis>
The query \"UK retirement age?\" has the following characteristics:
- Asks a single, straightforward question
- Seeks factual information about a specific topic
- Does not mention personal circumstances or conditional factors
- Does not have multiple parts
- Does not involve multiple stakeholders

This matches the criteria for a simple query. It's asking for basic factual information about the UK State Pension age without any personal circumstances that would affect the answer. I will use the `searchGovUk` tool.
</classification_analysis><answer_preparation>
**Relevant passages from search results:**

1. \"SPa is the earliest age at which people can receive their State Pension. Since October 2020, it has been set at 66 for both men and women who were born on or after 6 October 1954.\"

2. \"Current legislation provides for further increases to SPa in the following timescales:
- increase in SPa from 66 to 67 - between April 2026 and April 2028 (affecting those born on or after 6 April 1960)
- increase in SPa from 67 to 68 - between April 2044 and April 2046 (affecting those born on or after 6 April 1977)\"

3. \"The State Pension age is rising from 66 to 67 between April 2026 and March 2028. If you were born between 6 April 1960 and 5 March 1961, your State Pension age won't simply be 66; it could be 66 plus a number of additional months, depending on your exact date of birth.\"

**Sources to cite:**
- State Pension age Review 2023: a GAD technical bulletin (https://www.gov.uk/government/publications/state-pension-age-review-2023-a-gad-technical-bulletin/state-pension-age-review-2023-a-gad-technical-bulletin#background)
- LA Welfare Direct 3/2026 (https://www.gov.uk/government/publications/la-welfare-direct-bulletins-2026/la-welfare-direct-32026#why-it-matters)

**Answer structure plan:**
- Introductory sentence: State the current State Pension age
- Bullet point 1: Explain who this applies to (those born on or after 6 October 1954)
- Bullet point 2: Mention increase to 67 between April 2026 and April 2028 (those born on or after 6 April 1960)
- Bullet point 3: Note future increase to 68 between April 2044 and April 2046 (those born on or after 6 April 1977)
- Call to action: Direct them to check their specific State Pension age

**Bullet point count:** 3 bullet points - appropriate for a simple query
</answer_preparation>

```json
{
  \"answer\": \"The UK State Pension age is currently 66.\
\
- this applies to both men and women born on or after 6 October 1954\
- it will increase to 67 between April 2026 and April 2028 for those born on or after 6 April 1960\
- it will increase to 68 between April 2044 and April 2046 for those born on or after 6 April 1977\
\
[Check your State Pension age](https://www.gov.uk/state-pension-age) on GOV.UK by entering your date of birth.\",
  \"answered\": true,
  \"sources_used\": [\"https://www.gov.uk/government/publications/state-pension-age-review-2023-a-gad-technical-bulletin/state-pension-age-review-2023-a-gad-technical-bulletin#background\", \"https://www.gov.uk/government/publications/la-welfare-direct-bulletins-2026/la-welfare-direct-32026#why-it-matters\"]
}
```
"""
