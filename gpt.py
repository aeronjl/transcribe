from .openai import OpenAIClient

client = OpenAIClient()
openai = client.get_openai()

def process_transcription(chunk, system_prompt):
    completion = openai.chat.completions.create(
        model="gpt-4o",
        response_format={ "type" : "json_object" },
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": chunk
            }
        ],
        temperature=0.2
    )
    return completion