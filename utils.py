import os
from typing import List, Union
import ollama
import openai
import tiktoken
from dotenv import load_dotenv
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

client = Client(host='http://localhost:11434')
MAX_TOKENS_PER_CHUNK = (1000)


def num_tokens_in_string(
        input_str: str, encoding_name: str = "cl100k_base"
) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


def get_completion(
        prompt: str, system_message: str = "你是一位得力的助手。"
) -> Union[str, dict]:
    response = ollama.chat(
        model='qwen2',
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    )
    return response["message"]['content']


def singlechunk_translation(
        source_lang: str, target_lang: str, source_text: str
) -> str:
    system_message = f"您是一位语言专家，擅长从 {source_lang} 翻译成 {target_lang}."

    translation_prompt = f"""这是{source_lang}到{target_lang}的翻译，请提供此文本的{target_lang}翻译。\
    除翻译外，请勿提供任何解释或文字。
    {source_lang}: {source_text}

    {target_lang}:"""

    translation = get_completion(translation_prompt, system_message=system_message)

    return translation


def multichunk_translation(
        source_lang, target_lang, source_text_chunks
) -> List[str]:
    system_message = f"您是一位语言专家，擅长从 {source_lang} 翻译成 {target_lang}."

    translation_prompt = """您的任务是提供文本部分从 {source_lang} 到 {target_lang} 的专业翻译。

    源文本是{chunk_to_translate}，仅翻译源文本。

    仅输出您被要求翻译的部分的翻译，不要输出其他内容。
    """

    translation_chunks = []
    for i in range(len(source_text_chunks)):
        # Will translate chunk i
        print(source_text_chunks[i])
        print("----------------------------------------------")
        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = get_completion(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


def translate(
        source_lang,
        target_lang,
        source_text,
        max_tokens=MAX_TOKENS_PER_CHUNK,
):
    """Translate the source_text from source_lang to target_lang."""
    # 获取到所有文字
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as a single chunk")

        final_translation = singlechunk_translation(
            source_lang, target_lang, source_text
        )

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        translation_2_chunks = multichunk_translation(
            source_lang, target_lang, source_text_chunks
        )

        return "".join(translation_2_chunks)

