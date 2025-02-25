import time
import json
from typing import Optional

from .openai import OpenAIClient

client = OpenAIClient()
openai = client.get_openai()


def generate_system_prompt(speakers: Optional[int] = None) -> str:
    if speakers == None:
        speaker_prompt = "at least one respondent"
    elif speakers == 1:
        speaker_prompt = "one respondent"
    else:
        speaker_prompt = f"{speakers} respondents"

    system_prompt = f"""
    You are a helpful assistant whose job it is to label a transcript according to who is speaking.
    You will see a transcript from a conversation between an interviewer and {speaker_prompt}.
    
    IMPORTANT: Your primary goal is to preserve ALL content from the transcript. Do not skip, summarize, or condense any parts.
    
    Follow these steps:
    1. Reorganise and label the transcript so it is clear who is speaking
    2. Guess the name of the respondent from the context where possible
    3. Preserve ALL content - even if it seems unimportant or repetitive
    4. Remove only excessive filler words (um, uh, er) without changing the meaning or removing ANY content
    5. Only add necessary punctuation such as periods, commas, and capitalization
    
    CRITICAL: Every sentence and every exchange must be preserved in your output. Do not skip any section regardless of how trivial it may seem.
    
    You must return your response as a properly formatted JSON with a unique numbered key for each speaker turn.
    """

    examples = """
    \n\n
    --- Examples ---
    
    Example 1:

    Hi there, how are you, you know, doing today Emily? I'm um fine, thank you. Great. I'm going to show you some you know marketing materials. Is that okay? I mean, yes. No problem.

    {
        "1" : {
            "Speaker" : "Interviewer",
            "Content" : "Hi there, how are you doing today Emily?"
        },
        "2" : {
            "Speaker" : "Respondent 1 (Emily)",
            "Content" : "I'm fine, thank you!"
        },
        "3" : {
            "Speaker" : "Interviewer",
            "Content" : "Great. I'm going to show you some marketing materials. Is that okay?"
        }
        "4" : {
            "Speaker" : "Respondent 1 (Emily)",
            "Content" : "Yes. No problem."
        }
    }

    Example 2:

    It's hard because, you know, there are so many, um, things to consider. I mean, you know, it's not easy. I mean, it's not easy at all. I see. Thank you, Doctor. For our next exercise we're going to look at some headline statements. Take a look at these and tell me what you think.

    {
        "1" : {
            "Speaker" : "Respondent 1",
            "Content" : "It's hard because there are so many things to consider. It's not easy. It's not easy at all."
        },
        "2": {
            "Speaker" : "Interviewer",
            "Content" : "I see. Thank you, Doctor. For our next exercise we're going to look at some headline statements. Take a look at these and tell me what you think."
        }
    }
    
    Example 3 (notice how all content is preserved, even the partial repetitions):

    Well I think that, um, the key issue here is really about, you know, climate change. Climate change is the big one. And we have to address it now, we can't wait any longer to address it. We really can't. So what do you think about that perspective? I think you're right, climate change is the most important issue facing us today, and we need to take immediate action before it's too late.

    {
        "1" : {
            "Speaker" : "Respondent 1",
            "Content" : "Well I think that the key issue here is really about climate change. Climate change is the big one. And we have to address it now, we can't wait any longer to address it. We really can't."
        },
        "2" : {
            "Speaker" : "Interviewer",
            "Content" : "So what do you think about that perspective?"
        },
        "3" : {
            "Speaker" : "Respondent 2",
            "Content" : "I think you're right, climate change is the most important issue facing us today, and we need to take immediate action before it's too late."
        }
    }
    
    -- End of Examples --\n\n
    """

    system_prompt = system_prompt + examples

    return system_prompt


def process_transcription(chunk, system_prompt):

    t = time.time()
    completion = openai.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ],
        temperature=0.2,
    )
    elapsed = time.time() - t
    print(f"Processed transcription segment. Elapsed: {elapsed} seconds.")

    return completion


# Main execution
if __name__ == "__main__":
    pass
