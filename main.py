import json
import logging
import os
import re
import subprocess
from os import mkdir, makedirs

import json5

from pathlib import Path
from openai import OpenAI
from faster_whisper import WhisperModel, BatchedInferencePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)

ffmpeg_dir="D:/ffmpeg-6.1.1-essentials_build"
cache_dir = f".cache"

model_llama3 = "llama3.1:8b"
model_qwen25_7b = "qwen2.5:7b"
model_qwen25_14b = "qwen2.5:14b"
model=model_qwen25_14b

client = OpenAI(
    base_url='http://192.168.31.10:11434/v1',
    api_key='ollama',
)

def ollama_translate(sentences: list) -> list:
    json_text = json5.dumps(sentences)
    response = client.chat.completions.create(
        model=model,
        # temperature=0.5,
        messages=[
            {
                "role": "system",
                "content": """
                    You are a professional, real translation engine. You will be given input and steps to follow to translate the text without any explanation.
                    Please provide your answer in the following <output-format> and without any additional information.
                """
            },
            {
                "role": "user",
                "content": f"""
                     <input>
                         {json_text}
                     </input>
                     
                     <output-format>
                         [
                            {{
                                "text": "<<original text in english>>",
                                "step1": "<<translated text in chinese>>",
                                "step2": "<<refined translation in chinese>>"
                            }},
                            {{
                                "text": "<<original text in english>>",
                                "step1": "<<translated text in chinese>>",
                                "step2": "<<refined translation in chinese>>"
                            }},
                            ...
                         ]
                     </output-format>
                     
                     <steps>
                        1. Extract the content from the "text" field in the provided json object. Keep the original english text in the "text" field.
                        2. Translate the extracted content into chinese and place into the "step1" field.
                        3. Refine the initial translation from "step1" to make it more natural and understandable in chinese. Place this refined translation into the "step2" field.
                     </steps>
                """
            },
        ],

    )
    print(response.choices[0].message.content)
    return json5.loads(response.choices[0].message.content)


def ollama_segments_analysis(text: str) -> str:
    response = client.chat.completions.create(
        model=model,
        # temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": """
                    I have a video subtitle that needs to be split. You need to split the text given in the <input> tag according to punctuation or sentence meaning. 
                    The key is to ensure that each paragraph has less than 15 words, but it is more important to keep the semantic integrity of a single sentence.
                    Several examples are provided for your reference in <example>.
                    Please provide your answer in the following <output-format> and without any additional information.
                """
            },
            {
                "role": "user",
                "content": f"""
                    <input>
                        {text}
                    </input>
                    <output-format>
                        {{
                            "split": ["<<Output each sentence in sequence.>>"]
                        }}
                    </output-format>
                    <rules>
                        1. First segment the text at punctuation marks, such as period, comma, question mark, etc. (such as ",", ".", "?", "!" etc.)
                        2. You can also segment at conjunctions (such as "and", "but", "because", "when", "then", "if", "so", "that"). 
                        3. It can also be divided according to sentence structure, such as long inverted sentences, subordinate clauses, etc.
                        4. Output each sentence in sequence. For example, ["this is the first part,", "this is the second part."]
                    </rules>
                    <examples>
                        <example>
                            <input>
                                By now you know that I am the developer of the ConfUI IP Adapter extension and now also of the Instant ID one. Instant ID is a style transfer model targeted to people's portraits. Be careful because there are many Instant ID extensions in the manager, but mine is the only native one, meaning that it is fully
                            </input>
                            <output>
                                {{
                                    "split": ["By now you know that", "I am the developer of the ConfUI IP Adapter extension", "and now also of the Instant ID one.", "Instant ID is a style transfer model targeted to people's portraits.", "Be careful because there are many Instant ID extensions in the manager,", "but mine is the only native one,", "meaning that it is fully"]
                                }}
                            </output>
                        </example>
                        <example>
                            <input>
                                By now you know that I am the developer of the ConfUI IP Adapter extension and now also of the Instant ID one. Instant ID is a style transfer model targeted to people's portraits. Be careful because there are many Instant ID extensions in the manager, but mine is the only native one, meaning that it is fully
                            </input>
                            <output>
                                {{
                                    "split": ["By now you know that", "I am the developer of the ConfUI IP Adapter extension", "and now also of the Instant ID one.", "Instant ID is a style transfer model targeted to people's portraits.", "Be careful because there are many Instant ID extensions in the manager,", "but mine is the only native one,", "meaning that it is fully"]
                                }}
                            </output>
                        </example>
                    </examples>
                """.strip()
            },
        ],

    )
    logging.info(response.choices[0].message.content)
    return json5.loads(response.choices[0].message.content)

def get_whisper_segments(audio: str):
    # Run on GPU with FP16
    model = WhisperModel(
        model_size_or_path="./models/faster-whisper-large-v3",
        device="cuda",
        compute_type="float16",
        local_files_only=True,
        # num_workers=4,
    )

    batched_model = BatchedInferencePipeline(model=model, language="en", )
    segments, info = batched_model.transcribe(
        audio=audio,
        word_timestamps=True,
        batch_size=16,
        language="en",
        length_penalty=1,
        beam_size=5,
        # max_new_tokens=10,
        vad_filter=True,
        vad_parameters=dict(offset=0.363, min_silence_duration_ms=500),
    )

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    whisper_result = [
        {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                } for word in segment.words]
        } for segment in segments]
    return whisper_result

def load_or_execute(file_name, func, *args, **kwargs):
    """
    加载或创建JSON文件。

    参数:
    file_name (str): 文件名。
    func (callable): 闭包或方法，当文件不存在时调用。
    *args: 传递给闭包的 positional arguments。
    **kwargs: 传递给闭包的 keyword arguments。

    返回:
    dict: 文件内容或闭包的返回结果。
    """
    if os.path.exists(file_name):
        # 文件存在，读取并反序列化
        with open(file_name, 'r', encoding='utf-8') as file:
            return json5.load(file)
    else:
        # 文件不存在，执行闭包并保存结果
        result = func(*args, **kwargs)
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
        return result

# 定位每段的开始和结束
def find_sentences_times(sentences, words):
    word_index = 0
    for sentence in sentences:
        start_time = None
        end_time = None

        while word_index < len(words):
            word_time = words[word_index]
            word = word_time["word"].strip(" ,.!?")

            if word in sentence["text"]:
                if start_time is None:
                    start_time = word_time["start"]
                end_time = word_time["end"]

                word_index += 1
            elif start_time is None:
                word_index += 1
            else:
                break

        if start_time is not None and end_time is not None:
            sentence["start"] = start_time
            sentence["end"] = end_time
    return sentences

def seconds_to_srt_time(seconds):
    # 分离小时
    hours = int(seconds // 3600)
    # 剩余秒数
    seconds %= 3600
    # 分离分钟
    minutes = int(seconds // 60)
    # 剩余秒数
    seconds %= 60
    # 分离毫秒
    milliseconds = int((seconds - int(seconds)) * 100)
    # 格式化输出
    return f"{hours:01d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:02d}"

def write_subtitle(segments_analysis, subtitle_file):
    global index, analysis
    with open(f"{subtitle_file}", 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
; This is an Advanced Substation Alpha v4+ script.
Title: Untitled
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Chinese,SimHei,16,&H00FCD900,&H00000000,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,0,2,10,10,10,1
Style: English,Arial,12,&H00FFFFFF,&H00000000,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,4,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for index, analysis in enumerate(segments_analysis):
            for i, sub in enumerate(analysis['sentences']):
                start_time = seconds_to_srt_time(sub["start"])
                end_time = seconds_to_srt_time(sub["end"])
                english_text = sub["text"]
                chinese_text = sub["step2"]
                # 将标点符号后面追加 \N
                chinese_text = re.sub(r'([，。、！？）,.!?’])', '\\1 ', chinese_text)
                mid_index = len(chinese_text) // 2
                chinese_text = chinese_text[:mid_index] + ' ' + chinese_text[mid_index:] if len(chinese_text) > 35 and ' ' not in chinese_text else chinese_text

                f.write(f"Dialogue: 0,{start_time},{end_time},English,,0,0,0,,{english_text}\n")
                f.write(f"Dialogue: 0,{start_time},{end_time},Chinese,,0,0,0,,{chinese_text}\n")

def embed_subtitles(input_video, subtitle_file, output_video):
    # 构建 ffmpeg 命令
    command = [
        f'{ffmpeg_dir}/bin/ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', f'{input_video}',
        '-vf', f"ass=\\'{subtitle_file}\\'",
        '-threads', '24',
        '-c:v', 'h264_nvenc',
        '-c:a', 'copy',
        '-stats',
        f'{output_video}'
    ]

    # 调用 ffmpeg 命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in process.stderr:
        logging.info(line.strip())
    process.wait()
    logging.info("ffmpeg process completed.")

def process_video(input_video):
    makedirs(name=f"{cache_dir}", mode=755, exist_ok=True)
    subtitle_file = f"{cache_dir}/subtitle.ass"

    whisper_result = load_or_execute(f"{cache_dir}/whisper_result.json", get_whisper_segments, input_video)
    segments_analysis = load_or_execute(f"{cache_dir}/ollama_segments_analysis.json", lambda rs: [{**segment, **ollama_segments_analysis(segment["text"])} for segment in rs], whisper_result)

    for index, analysis in enumerate(segments_analysis):
        analysis['sentences'] = [{"text": text.strip(), "step1": "", "step2": ""} for text in analysis["split"]]

    segments_analysis = load_or_execute(f"{cache_dir}/ollama_translate.json", lambda objs: [{**segment, "sentences": ollama_translate(segment["sentences"])} for segment in objs], segments_analysis)

    for index, analysis in enumerate(segments_analysis):
        analysis['sentences']=find_sentences_times(analysis['sentences'], analysis["words"])

    write_subtitle(segments_analysis, subtitle_file)

    embed_subtitles(Path(input_video).absolute(), Path(subtitle_file).absolute().as_posix(), rf"{Path(cache_dir).absolute()}\output_video.mp4")

if __name__ == "__main__":
    file_path = r"g:\Downloads\How Elon Musk found his way into Trump's inner circle [j1A12n8thHY].webm"
    process_video(file_path)