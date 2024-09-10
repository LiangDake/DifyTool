import os
import random
from py2neo import Graph
import requests

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import module
from module import *

app = Flask(__name__)

# 上传单个文件
BASE_UPLOAD_FOLDER = 'static/uploads/'

app.config['BASE_UPLOAD_FOLDER'] = BASE_UPLOAD_FOLDER

# 上传知识库
BASE_KNOWLEDGE_FOLDER = 'static/knowledge/'

app.config['BASE_KNOWLEDGE_FOLDER'] = BASE_KNOWLEDGE_FOLDER

# Set the folder where the files will be saved
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# @app.route('/')
# def main():
#     # Get the list of folders in the BASE_UPLOAD_FOLDER
#     folders = os.listdir(app.config['BASE_UPLOAD_FOLDER'])
#     folders = [f for f in folders if os.path.isdir(os.path.join(app.config['BASE_UPLOAD_FOLDER'], f))]
#
#     # Get files in each folder
#     folder_files = {}
#     for folder in folders:
#         folder_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], folder)
#         folder_files[folder] = os.listdir(folder_path)
#
#     return jsonify(folder_files)


# 上传文件至文件夹
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # 获取到文件夹名
        folder_name = request.form.get('folder', 'Default').strip()
        if not folder_name:
            return jsonify({"error": "Folder name is required."}), 400

        # Create the full folder path
        full_folder_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], folder_name)
        os.makedirs(full_folder_path, exist_ok=True)

        # 获取到所有选择文件
        files = request.files.getlist("file")

        # Iterate for each file in the files List, and Save them
        for file in files:
            file.save(os.path.join(full_folder_path, file.filename))

        return jsonify({"message": "Files Uploaded Successfully!"})


# 创建知识库
@app.route('/build_knowledge_base', methods=['POST'])
def build_knowledge_base():
    # 获取用户输入的folder_name, query_text 和 chroma_path 参数
    folder_name = request.form.get('folder', '').strip()
    chroma_path = request.args.get('chroma_path', 'default').strip()

    if not folder_name:
        return jsonify({"error": "Folder name is required."}), 400

    # Ensure the folder exists
    full_folder_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], folder_name)
    if not os.path.exists(full_folder_path) or not os.path.isdir(full_folder_path):
        return jsonify({"error": "Folder does not exist."}), 404

    # Get files in the selected folder
    files = os.listdir(full_folder_path)

    # Instantiate the KnowledgeBase class
    knowledge_base_instance = KnowledgeBase(chroma_path=chroma_path)

    result = knowledge_base_instance.add_folder(folder_path=full_folder_path)

    return jsonify({
        "folder_name": folder_name,
        "files": files,
        "response": result
    })


# 查询知识库
@app.route('/query_knowledge_base', methods=['POST'])
def query_knowledge_base():
    # 获取用户输入的参数
    query_text = request.args.get('query_text', '').strip()
    chroma_path = request.args.get('chroma_path', 'default').strip()
    file_num = request.args.get('file_num', '').strip()

    if not chroma_path:
        return jsonify({"error": "Knowledge name is required."}), 400

    # Ensure the folder exists
    full_knowledger_path = os.path.join(app.config['BASE_KNOWLEDGE_FOLDER'], chroma_path)
    if not os.path.exists(full_knowledger_path):
        return jsonify({"error": "Knowledge does not exist."}), 404

    query = QueryKnowledgeBase(chroma_path=chroma_path, file_num=file_num)

    response = query.query_content(query_text=query_text)

    return jsonify({
        "query_text": query_text,
        "response": response
    })


# 翻译上传页面
@app.route('/translate_upload')
def translate_form():
    return render_template('translate_upload.html')


# 翻译（跳转页面式）
@app.route('/translate', methods=['POST'])
def translate():
    # Check if a file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # If the user does not select a file, the browser may submit an empty part
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get user-specified parameters from the request
    source_lang = request.form.get('source_lang', 'English').strip()
    target_lang = request.form.get('target_lang', 'Chinese').strip()

    if file:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Translate the content of the saved file
        query_file = QueryFileBase()
        result = query_file.translate_content(file_path=file_path, source_lang=source_lang, target_lang=target_lang)
        # Optionally remove the file after translation
        os.remove(file_path)

        return result


# # 翻译
# @app.route('/translate', methods=['POST'])
# def translate():
#     file_url = request.json.get('file_url', '').strip()
#     source_lang = request.json.get('source_lang', 'English').strip()
#     target_lang = request.json.get('target_lang', 'Chinese').strip()
#
#     if not file_url:
#         return jsonify({"error": "file_url is required"}), 400
#
#     try:
#         print(file_url)
#         # Download the file from the URL
#         response = requests.get(file_url)
#         response.raise_for_status()
#
#         # Secure the filename and save the file
#         filename = secure_filename(file_url.split('/')[-1])
#         file_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], filename)
#
#         with open(file_path, 'wb') as f:
#             f.write(response.content)
#
#         # Translate the content of the saved file
#         query_file = QueryFileBase()
#         result = query_file.translate_content(file_path=file_path, source_lang=source_lang, target_lang=target_lang)
#
#         # Optionally remove the file after translation
#         os.remove(file_path)
#
#         return result
#
#     except requests.exceptions.RequestException as e:
#         return jsonify({"error": str(e)}), 500


# 总结上传页面
@app.route('/conclusion_upload')
def upload_form():
    return render_template('conclusion_upload.html')


# 总结（跳转页面式）
@app.route('/conclusion', methods=['POST'])
def conclusion():
    if 'files' not in request.files:
        return "No files part"

    files = request.files.getlist('files')  # Get the list of files

    file_paths = []  # 保存文件路径以便后续删除
    for file in files:
        if file and file.filename:  # 确保文件有名字
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            file_paths.append(file_path)

    # Here you can add your analysis code for the uploaded files
    result = QueryFileBase().conclusion(folder_path=UPLOAD_FOLDER)

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    return result


# # 总结
# @app.route('/conclusion', methods=['POST'])
# def conclusion():
#     # 从请求中获取文件 URL 列表
#     file_urls = request.json.get('file_urls', [])
#
#     if not file_urls:
#         return jsonify({"error": "No file URLs provided"}), 400
#
#     file_paths = []  # 保存文件路径以便后续删除
#     for file_url in file_urls:
#         try:
#             # 下载文件
#             response = requests.get(file_url)
#             response.raise_for_status()
#
#             # 确保文件名安全
#             filename = secure_filename(file_url.split('/')[-1])
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#
#             # 保存文件到指定文件夹
#             with open(file_path, 'wb') as f:
#                 f.write(response.content)
#
#             file_paths.append(file_path)
#         except requests.exceptions.RequestException as e:
#             return jsonify({"error": f"Failed to download file from {file_url}. Reason: {str(e)}"}), 500
#
#     # 对所有上传文件的分析总结
#     result = QueryFileBase().conclusion(folder_path=app.config['UPLOAD_FOLDER'])
#
#     # 删除已保存的文件
#     for file_path in file_paths:
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')
#
#     return result


@app.route('/build_knowledge_graph', methods=['POST'])
def build_knowledge_graph():
    # 从请求中获取文件 URL 列表
    file_urls = request.json.get('file_urls', [])

    if not file_urls:
        return jsonify({"error": "No file URLs provided"}), 400

    file_paths = []  # 保存文件路径以便后续删除
    for file_url in file_urls:
        try:
            # 下载文件
            response = requests.get(file_url)
            response.raise_for_status()

            # 确保文件名安全
            filename = secure_filename(file_url.split('/')[-1])
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 保存文件到指定文件夹
            with open(file_path, 'wb') as f:
                f.write(response.content)

            file_paths.append(file_path)
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to download file from {file_url}. Reason: {str(e)}"}, 500

    # 初始化图知识库
    graph_base = module.GraphBase()
    # 构建知识图谱
    result = graph_base.build_knowledge_graph(folder_path=app.config['UPLOAD_FOLDER'])
    if result is False:
        return {f"{file_urls}上传失败，请检查URL链接中的文件是否符合文件类型要求。"}
    else:
        return {f"{file_urls} 中的文件已成功上传并成功构建知识图谱，请在您的Neo4j知识库中查看。"}


@app.route('/query_knowledge_graph', methods=['POST'])
def query_knowledge_graph():
    query = request.json.get('query', '').strip()
    # 确定图知识库
    graph_base = module.GraphBase()
    # 获取结果
    try:
        final_answer = graph_base.query_knowleage_graph(query=query)
        return final_answer
    except Exception as e:
        return jsonify({f"查询失败，原因如下：{e}"})


@app.route('/delete_knowledge_graph', methods=['GET'])
def delete_knowledge_graph():
    graph_base = module.GraphBase()
    # 删除知识图谱
    result = graph_base.delete_knowleage_graph()
    if result is False:
        return "删除失败，请检查网络设置", 500
    else:
        return "删除成功！", 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=4396)