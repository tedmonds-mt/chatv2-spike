from invoke import extract_answer
import pytest

test_str1 = "abc"
test_str2 = "<classification_analysis>abc</classification_analysis>def"
test_str3 = """
<classification_analysis>
The user is asking about my role or function. This is a meta-question about what I am and what I can help with, rather
than a question about UK government services or information.

This doesn't fit the pattern of either a simple or complex query about government services. It's a question about me as 
an assistant. I should answer this directly without using the search tools, as no search is needed to explain my
purpose.
</classification_analysis>
I'm a professional government assistant designed to help you find accurate information about UK government services.
I can help you with questions about:
- benefits and financial support
- taxes and Self Assessment
- passports, visas, and immigration
- driving and transport
- business and employment
- education and skills
- housing and local services
- many other government topics
I search official GOV.UK guidance to give you clear, reliable answers based on the latest government information.

## How to get help
Ask me a specific question about a UK government service or topic, and I'll find the relevant information for you.
For example, you could ask "How do I renew my passport?" or "What benefits can I claim if I'm unemployed?"
"""

test_str4 = (
    '<classification_analysis>\nThe query "What do you do?" is asking about my role and capabilities as '
    "a government assistant. This is a simple, straightforward question about:\n- It asks a single "
    "question\n- It doesn't involve personal circumstances or conditional factors\n- It seeks factual "
    "information about what I can help with\n- It's a meta-query about the service itself\n\nHowever, "
    "this is not actually a query about UK government services or policies - it's a question about me as an "
    "assistant. This doesn't require searching GOV.UK or using the complex_search tool. I should answer "
    "directly based on my instructions about my role.\n</classification_analysis>\n\n<answer_preparation>\n"
    "This query is asking about my role and function. I don't need to search for information since this is "
    "about my capabilities as defined in my instructions. I should:\n1. Briefly explain my purpose\n2. List "
    "the types of questions I can help with\n3. Encourage them to ask a question\n\nStructure:\n- Opening "
    "sentence explaining I'm a government assistant\n- 2-3 bullets about what I help with\n- Call to action "
    "inviting them to ask a question\n\nNo external sources needed as this is about my role.\n"
    '</answer_preparation>\n\n```json\n{\n  "answer": "I\'m a UK government assistant that helps you find '
    "accurate, official information about government services.\\n\\nI can help you with questions about:\\n\\n"
    "- benefits and financial support\\n- tax and Self Assessment\\n- visas and immigration\\n- driving and "
    "transport\\n- business and employment\\n- many other government services\\n\\nWhat would you like to "
    'know about UK government services?",\n  "answered": true,\n  "sources_used": []\n}\n```\n'
)


@pytest.mark.describe("invocation code")
class TestInvokeExtractAnswer(object):
    @pytest.mark.it("returns a string")
    def test_returns_a_string(self):
        assert type(extract_answer(test_str1)) is str

    @pytest.mark.it("leaves any string without classification unchanged")
    def test_leaves_any_string_without_classification(self):
        assert extract_answer(test_str1) == "abc"

    @pytest.mark.it("returns only the part outside the classification")
    def test_return_only_relevant_part_outside_html(self):
        result = extract_answer(test_str3)
        assert result[0:10] == "I'm a prof"
        assert result[-12:] == 'unemployed?"'

    @pytest.mark.it("returns only the answer in an embedded json string")
    def test_returns_embedded_json_answer(self):
        result = extract_answer(test_str4)
        assert result[0:19] == "I'm a UK government"
        assert result[-9:] == "services?"
