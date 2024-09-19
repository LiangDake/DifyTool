import email
import zipfile
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
import os

from werkzeug.utils import secure_filename

import base
import file_processing

app = Flask(__name__)

# 已上传文件夹
UPLOAD_FOLDER = 'uploads'

# 已翻译文件夹
PROCESSED_FOLDER = 'processed'

# 已构建知识图谱
GRAPH_FOLDER = 'graph'
# 知识库文件夹
BASE_KNOWLEDGE_FOLDER = 'static/knowledge/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['BASE_KNOWLEDGE_FOLDER'] = BASE_KNOWLEDGE_FOLDER



# 添加路由，允许访问上传的文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# 添加路由，允许访问处理过的文件
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


# 上传文件并生成URL链接
@app.route('/upload', methods=['POST'])
def upload_files_and_generate_links():
    global uploaded_files
    if 'files' not in request.files:
        return redirect(request.url)

    # 获取上传的文件
    files = request.files.getlist('files')

    file_links = []  # 用于存储文件链接

    # 遍历每一个文件
    for file in files:
        if file.filename == '':
            continue

        # 使用 secure_filename 确保文件名安全并正确
        filename = secure_filename(file.filename)
        # 获取到文件路径
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)
        # 添加文件链接
        file_link = url_for('uploaded_file', filename=filename, _external=True)
        file_links.append(file_link)

        if filename not in uploaded_files:
            uploaded_files.append(filename)

    # return redirect(url_for('index'))
    # 返回文件链接列表给前端
    return jsonify({"uploaded_files": file_links}), 200


# 首页显示上传和已处理的文件
@app.route('/')
def index():
    # 获取已上传文件夹中的文件
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])

    # 获取处理文件夹中的文件
    processed_files = os.listdir(app.config['PROCESSED_FOLDER'])

    # 获取 BASE_UPLOAD_FOLDER 中的所有子文件夹
    knowledge_folders = [f for f in os.listdir(app.config['BASE_KNOWLEDGE_FOLDER']) if
                         os.path.isdir(os.path.join(app.config['BASE_KNOWLEDGE_FOLDER'], f))]

    return render_template('index.html',
                           uploaded_files=uploaded_files,
                           processed_files=processed_files,
                           knowledge_folders=knowledge_folders)


# 预览已上传文件
@app.route('/preview/<filename>')
def preview_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# 删除已上传文件
@app.route('/delete_uploaded/<filename>', methods=['POST'])
def delete_uploaded_file(filename):
    global uploaded_files
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(filepath):
        os.remove(filepath)
        uploaded_files.remove(filename)

    return redirect(url_for('index'))


# 翻译已上传文件
@app.route('/translate', methods=['POST'])
def translate_files():
    # 获取请求中的文件链接列表
    data = request.get_json()
    file_links = data.get('file_links', [])

    if not file_links:
        return jsonify({"error": "No file links provided"}), 400

    translated_files = []

    for file_link in file_links:
        try:
            # 下载文件并获得对应的本地路径
            file_path = file_processing.download_file_from_url(file_link, 'static/download')
            # 获得文件链接
            file_type = file_processing.get_file_type(file_path)
            # 判断是否为email
            if file_type in (".eml", ".msg"):
                # 调用邮件翻译功能，所有翻译后的文件均保存至processed文件夹
                translated_file_path = base.QueryFileBase().translate_email(
                    file_path=file_path
                )
            # 判断是否为支持翻译的文件类型
            elif file_processing.is_supported_formats(file_path):
                # 调用文件翻译功能，所有翻译后的文件均保存至processed文件夹
                translated_file_path = base.QueryFileBase().translate_file(
                    file_path=file_path
                )
            # 如果不是支持翻译的文件类型，则继续下一个
            else:
                continue
            # 构建翻译后文件的链接
            translated_file_link = url_for('processed_file', filename=os.path.basename(translated_file_path), _external=True)

            # 记录翻译后的文件链接
            translated_files.append(translated_file_link)

        except Exception as e:
            return jsonify({"error": f"Failed to translate {file_link}: {str(e)}"}), 500

    # 返回翻译后的文件链接
    return jsonify({"translated_files": translated_files}), 200


# 总结已上传文件
@app.route('/conclusion', methods=['POST'])
def conclude_files():
    # 对所有上传文件的分析总结
    concluded_result = base.QueryFileBase().conclusion(folder_path=app.config['UPLOAD_FOLDER'])

    return redirect(url_for('index'))


# 构建已上传文件图谱
@app.route('/build_knowledge_graph', methods=['POST'])
def build_knowledge_graph():
    # 初始化图知识库（图形数据库需要更改URL地址与账号密码）
    graph_base = base.GraphBase()

    # 获取请求中的文件链接列表
    data = request.get_json()
    file_links = data.get('file_links', [])

    if not file_links:
        return jsonify({"error": "No file links provided"}), 400

    for file_link in file_links:
        try:
            # 下载文件并获得对应的本地路径
            file_path = file_processing.download_file_from_url(file_link, 'static/download')
            # 判断是否为支持上传的文件类型
            if file_processing.is_supported_formats(file_path):
                # 构建知识图谱（先翻译再构建）
                result = graph_base.build_knowledge_graph(file_path)
            # 如果不是支持上传的文件类型，则继续下一个
            else:
                continue
        except Exception as e:
            return jsonify({"error": f"Failed to upload {file_link}: {str(e)}"}), 500

    return jsonify({"summary_success": True})

# 删除已上传文件图谱
@app.route('/delete_knowledge_graph', methods=['POST'])
def delete_knowledge_graph():
    graph_base = base.GraphBase()
    # 删除知识图谱
    result = graph_base.delete_knowleage_graph()
    if result is True:
        return jsonify({"delete_success": True})
    else:
        return jsonify({"delete_success": False})


# 知识库

# 查询知识库
@app.route('/query_knowledge_base', methods=['POST'])
def query_knowledge_base():
    # 获取用户输入的参数
    query_text = request.form.get('query_text', '').strip()
    chroma_path = request.form.get('chroma_path', 'default').strip()
    file_num = request.form.get('file_num', '').strip()
    file_num = int(file_num)
    if not chroma_path:
        return jsonify({"error": "Knowledge name is required."}), 400

    # Ensure the folder exists
    full_knowledger_path = os.path.join(app.config['BASE_KNOWLEDGE_FOLDER'], chroma_path)
    if not os.path.exists(full_knowledger_path):
        return jsonify({"error": "Knowledge does not exist."}), 404

    query = base.QueryKnowledgeBase(chroma_path=chroma_path, file_num=file_num)

    response = query.query_content(query_text=query_text)

    return jsonify({
        "query_text": query_text,
        "response": response
    })


# 查询知识库
@app.route('/query_knowledge_base', methods=['POST'])
def query_knowledge_base():
    # 获取用户输入的参数
    query_text = request.form.get('query_text', '').strip()
    chroma_path = request.form.get('chroma_path', 'default').strip()
    file_num = request.form.get('file_num', '').strip()
    file_num = int(file_num)
    if not chroma_path:
        return jsonify({"error": "Knowledge name is required."}), 400

    # Ensure the folder exists
    full_knowledger_path = os.path.join(app.config['BASE_KNOWLEDGE_FOLDER'], chroma_path)
    if not os.path.exists(full_knowledger_path):
        return jsonify({"error": "Knowledge does not exist."}), 404

    query = base.QueryKnowledgeBase(chroma_path=chroma_path, file_num=file_num)

    response = query.query_content(query_text=query_text)

    return jsonify({
        "query_text": query_text,
        "response": response
    })


# 智能客服知识查询
@app.route('/customer_query', methods=['POST'])
def query_knowledge_base():
    # 获取用户输入的参数
    query_text = request.form.get('query_text', '').strip()

    # 检索知识库
    full_knowledge_path = os.path.join(app.config['BASE_KNOWLEDGE_FOLDER'], 'image')
    if not os.path.exists(full_knowledge_path):
        return jsonify({"error": "Knowledge does not exist."}), 404

    query = base.QueryKnowledgeBase(chroma_path=full_knowledge_path, file_num=10)

    response = query.query_content(query_text=query_text)

    return jsonify({
        "query_text": query_text,
        "response": response
    })


# 删除已处理文件
@app.route('/delete_processed/<filename>', methods=['POST'])
def delete_processed_file(filename):
    global uploaded_files
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    return redirect(url_for('index'))


# 下载已处理文件
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=4398)

