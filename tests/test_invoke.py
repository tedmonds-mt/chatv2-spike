import pytest

from invoke import Extractor


@pytest.fixture
def extractor():
    return Extractor()


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
    'know about UK government services?",\n  "answered": true,\n  "sources_used": []\n}\n```\n hello'
)

chunk1 = (
    '<classification_analysis>\nThe query "What do you do?" is asking about my role and capabilities as '
    "a government assistant. This is a simple, straightforward question about:\n- It asks a single "
    "question\n- It doesn't involve personal circumstances or conditional factors\n- It seeks factual "
    "information about what I can help with\n- It's a meta-query about the service itself\n\nHowever, "
)

chunk2 = (
    "this is not actually a query about UK government services or policies - it's a question about me as an "
    "assistant. This doesn't require searching GOV.UK or using the complex_search tool. I should answer "
    "directly based on my instructions about my role.\n</classification_analysis>\n\n<answer_preparation>\n"
    "This query is asking about my role and function. I don't need to search for information since this is "
    "about my capabilities as defined in my instructions. I should:\n1. Briefly explain my purpose\n2. List "
    "the types of questions I can help with\n3. Encourage them to ask a question\n\nStructure:\n- Opening "
    "sentence explaining I'm a government assistant\n- 2-3 bullets about what I help with\n- Call to action "
)

chunk3 = (
    "the types of questions I can help with\n3. Encourage them to ask a question\n\nStructure:\n- Opening "
    "sentence explaining I'm a government assistant\n- 2-3 bullets about what I help with\n- Call to action "
    "inviting them to ask a question\n\nNo external sources needed as this is about my role.\n"
    '</answer_preparation>\n\n```json\n{\n  "answer": "I\'m a UK government assistant that helps you find '
    "accurate, official information about government services.\\n\\nI can help you with questions about:\\n\\n"
)

chunk4 = (
    "- benefits and financial support\\n- tax and Self Assessment\\n- visas and immigration\\n- driving and "
    "transport\\n- business and employment\\n- many other government services\\n\\nWhat would you like to "
    'know about UK government services?",\n  "answered": true,\n  "sources_used": []\n}\n```\n'
)

dnl = (
    "the types of questions I can help with\n3. Encourage them to ask a question\n\nStructure:\\n- Opening "
    "sentence explaining I'm a government assistant\n- 2-3 bullets about what I help with\n- Call to action "
    "inviting them to ask a question\\nNo external sources needed as this is about my role.\n"
    '</answer_preparation>\n\\n```json\n{\n  "answer": "I\'m a UK government assistant that helps you find '
    "accurate, official information about government services.\n\nI can help you with questions about:\n"
)

dnl_expected = (
    "the types of questions I can help with\n3. Encourage them to ask a question\n\nStructure:\n- Opening "
    "sentence explaining I'm a government assistant\n- 2-3 bullets about what I help with\n- Call to action "
    "inviting them to ask a question\nNo external sources needed as this is about my role.\n"
    '</answer_preparation>\n\n```json\n{\n  "answer": "I\'m a UK government assistant that helps you find '
    "accurate, official information about government services.\n\nI can help you with questions about:\n"
)


@pytest.mark.describe("invocation code")
class TestInvokeExtractAnswer(object):
    @pytest.mark.it("returns a string")
    def test_returns_a_string(self, extractor):
        assert type(extractor.extract_answer(test_str1)) is str

    @pytest.mark.skip
    @pytest.mark.it("leaves any string without classification unchanged")
    def test_leaves_any_string_without_classification(self, extractor):
        assert extractor.extract_answer(test_str1) == "abc"

    @pytest.mark.skip
    @pytest.mark.it("returns only the part outside the classification")
    def test_return_only_relevant_part_outside_html(self, extractor):
        result = extractor.extract_answer(test_str3)
        assert result[0:10] == "I'm a prof"
        assert result[-12:] == 'unemployed?"'

    @pytest.mark.it("returns only the answer in an embedded json string")
    def test_returns_embedded_json_answer(self, extractor):
        result = extractor.extract_answer(test_str4)
        assert result[0:19] == "I'm a UK government"
        assert result[-9:] == "services?"

    @pytest.mark.it("deals with multichunk input")
    def test_multichunk_input(self, extractor):
        chunks = [chunk1, chunk2, chunk3, chunk4]
        result = ""
        for chunk in chunks:
            result = extractor.extract_answer(chunk)
        assert result[0:19] == "I'm a UK government"
        assert result[-9:] == "services?"

    @pytest.mark.it("has a cleaner to remove double newlines from final output")
    @pytest.mark.parametrize(
        "input_str, expected_output",
        [
            # Case 1: Correctly formatted newline left alone
            ("Hello\\n\nWorld", "Hello\n\nWorld"),
            # Case 2: escaped newline changed
            ("Line 1\\nLine 2", "Line 1\nLine 2"),
            # Case 3: No change needed for single newlines
            ("Line 1\nLine 2\nLine 3", "Line 1\nLine 2\nLine 3"),
            # Case 4: Mixture of different counts
            ("A\\nB\\n\\nC\nD", "A\nB\n\nC\nD"),
            # Case 5: Empty string
            ("", ""),
            # Case 6: String with only newlines
            ("\\n\\n\\n", "\n\n\n"),
            # Case 7: Newlines at the start and end
            ("\\nStart\\nEnd\\n", "\nStart\nEnd\n"),
            # Case 8: No newlines at all
            ("Plain text", "Plain text"),
            # Case 9: realistic case
            (dnl, dnl_expected),
            (
                "The State Pension age depends on when you were born.\\n- the current State Pension age is 66 for people born on or after 6 October 1954\n- it will rise to 67 between April 2026 and April 2028 for people born on or after 6 April 1960\n- it will rise to 68 between April 2044 and April 2046 for people born on or after 6 April 1977\\nYou can [check your exact State Pension age on GOV.UK](https://www.gov.uk/state-pension-age) by entering your date of birth.",
                "The State Pension age depends on when you were born.\n- the current State Pension age is 66 for people born on or after 6 October 1954\n- it will rise to 67 between April 2026 and April 2028 for people born on or after 6 April 1960\n- it will rise to 68 between April 2044 and April 2046 for people born on or after 6 April 1977\nYou can [check your exact State Pension age on GOV.UK](https://www.gov.uk/state-pension-age) by entering your date of birth.",
            ),
        ],
    )
    def test_simplify_newlines(self, input_str, expected_output):
        assert Extractor.clean_newlines(input_str) == expected_output

    @pytest.mark.it("Thinking should be returned")
    @pytest.mark.parametrize(
        "input_str, expected_output",
        [
            (
                "<thinking>Thought on one line</thinking>",
                "<small><i>Thinking (0/5): Thought on one line</i></small>",
            ),
            (
                "I am some preamble to be ignored<thinking>Thought on one line</thinking>",
                "<small><i>Thinking (0/5): Thought on one line</i></small>",
            ),
            ("No thoughts head empty", ""),
            ("I am in a thinking block with no opener</thinking>", ""),
            (
                "<thinking>\nI \n am \n a \n thought\n over multiple lines</thinking>",
                "<small><i>Thinking (0/5): \nI \n am \n a \n thought\n over multiple lines</i></small>",
            ),
            (
                "<thinking> First thought to be ignored</thinking><thinking>Return only the most recent thought</thinking>",
                "<small><i>Thinking (0/5): Return only the most recent thought</i></small>",
            ),
            (
                "<thinking>I am an unfinished thought",
                "<small><i>Thinking (0/5): I am an unfinished thought</i></small>",
            ),
            (
                '<thinking>Ignores if json answer</thinking>```json{"answer": "text',
                "text",
            ),
            (
                "<thinking>Thought with count<count>1</count></thinking>",
                "<small><i>Thinking (4/5): Thought with count</i></small>",
            ),
            (
                "<thinking>Remove reflection <reflection>tags</reflection></thinking>",
                "<small><i>Thinking (0/5): Remove reflection tags</i></small>",
            ),
        ],
    )
    def test_shows_most_recent_thoughts(self, input_str, expected_output, extractor):
        assert extractor.extract_answer(input_str) == expected_output
