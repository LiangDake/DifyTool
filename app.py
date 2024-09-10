import email
import zipfile
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
import os

from werkzeug.utils import secure_filename

import module

app = Flask(__name__)

# 文件夹配置
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
# 知识库文件夹
BASE_KNOWLEDGE_FOLDER = 'static/knowledge/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['BASE_KNOWLEDGE_FOLDER'] = BASE_KNOWLEDGE_FOLDER

# 已上传文件列表
uploaded_files = []


# 启动时加载上传文件夹中的文件
def load_uploaded_files():
    global uploaded_files
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])


def process_eml_file(filepath):
    # 使用不同的变量名来打开文件，避免变量冲突
    with open(filepath, 'rb') as eml_file:
        msg = email.message_from_binary_file(eml_file)

    # 遍历邮件的各个部分
    for part in msg.walk():
        # 如果邮件内容是附件
        if part.get_content_maintype() == "multipart":
            continue
        if part.get("Content-Disposition") is None:
            continue

        # 获取附件文件名
        filename = part.get_filename()
        if filename:
            # 解码附件文件名并确保安全
            filename = email.header.decode_header(filename)[0][0]
            if isinstance(filename, bytes):
                filename = filename.decode()
            filename = secure_filename(filename)  # 确保文件名安全

            # 保存附件到指定文件夹
            attachment_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(attachment_path, "wb") as f:
                f.write(part.get_payload(decode=True))

            # 添加到已上传文件列表
            if filename not in uploaded_files:
                uploaded_files.append(filename)


# 添加路由，允许访问上传的文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# 添加路由，允许访问上传的文件
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_files():
    global uploaded_files
    if 'files' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files')
    file_links = []  # 用于存储文件链接

    # 遍历每一个文件
    for uploaded_file in files:
        if uploaded_file.filename == '':
            continue

        # 使用 secure_filename 确保文件名安全
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # 判断是否为zip
        if filename.endswith('.zip'):
            # 如果是 .zip 文件，解压缩并上传其中的所有文件，平铺所有文件
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # 过滤掉无关的 Mac 系统文件和文件夹
                    if file_info.is_dir() or '__MACOSX' in file_info.filename or file_info.filename.endswith(
                            '.DS_Store'):
                        continue

                    # 获取文件的基本文件名，去掉父文件夹路径，并确保安全
                    extracted_filename = secure_filename(os.path.basename(file_info.filename))
                    extracted_path = os.path.join(app.config['UPLOAD_FOLDER'], extracted_filename)

                    # 保存文件
                    with zip_ref.open(file_info) as source, open(extracted_path, 'wb') as target:
                        target.write(source.read())

                    # 如果解压出来的是 .eml 文件，处理该文件
                    if extracted_filename.endswith('.eml'):
                        process_eml_file(extracted_path)

                    # 添加文件链接
                    file_link = url_for('uploaded_file', filename=extracted_filename, _external=True)
                    file_links.append(file_link)

                    # 添加到已上传文件列表
                    if extracted_filename not in uploaded_files:
                        uploaded_files.append(extracted_filename)

        # 判断是否为eml
        elif filename.endswith('.eml'):
            # 处理 .eml 文件，提取附件并上传
            uploaded_file.save(filepath)  # 先保存 .eml 文件
            process_eml_file(filepath)  # 处理附件

            # 添加文件链接
            file_link = url_for('uploaded_file', filename=filename, _external=True)
            file_links.append(file_link)

        # 保存其他文件
        else:
            uploaded_file.save(filepath)
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


# # 翻译已上传文件
# @app.route('/translate', methods=['POST'])
# def translate_files():
#     for filename in uploaded_files:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         # 翻译单个文件
#         translated_content = module.QueryFileBase().translate_content(file_path=filepath, source_lang="English",
#                                                                       target_lang="Chinese")
#         # 保存单个文件
#         save_translated_file(filepath, translated_content)
#
#     return redirect(url_for('index'))

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
            # 获取文件名并找到对应的本地路径
            filename = os.path.basename(file_link)
            if filename not in uploaded_files:
                return jsonify({"error": f"File {filename} not found in uploaded files"}), 404

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 调用翻译功能（使用你现有的翻译模块）
            translated_content = module.QueryFileBase().translate_content(
                file_path=filepath,
                source_lang="English",
                target_lang="Chinese"
            )

            # 保存翻译后的文件并获取保存路径
            translated_filepath = save_translated_file(filepath, translated_content)

            # 构建翻译后文件的链接
            translated_file_link = url_for('processed_file', filename=os.path.basename(translated_filepath), _external=True)

            # 记录翻译后的文件链接
            translated_files.append(translated_file_link)

        except Exception as e:
            return jsonify({"error": f"Failed to translate {filename}: {str(e)}"}), 500

    # 返回翻译后的文件链接
    return jsonify({"translated_files": translated_files}), 200


# 保存已翻译文件
def save_translated_file(filepath, translated_content):
    filename = os.path.basename(filepath)
    txt_filename = os.path.splitext(filename)[0] + '_translated.txt'
    txt_filepath = os.path.join(app.config['PROCESSED_FOLDER'], txt_filename)

    with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(translated_content)

    # 返回翻译后文件的路径
    return txt_filepath


# 总结已上传文件
@app.route('/conclusion', methods=['POST'])
def conclude_files():
    # 对所有上传文件的分析总结
    concluded_result = module.QueryFileBase().conclusion(folder_path=app.config['UPLOAD_FOLDER'])
    # 保存总结后的文本文件
    save_concluded_file(app.config['UPLOAD_FOLDER'], concluded_result)

    return redirect(url_for('index'))


# 保存已总结文件
def save_concluded_file(folderpath, translated_content):
    txt_filename = folderpath + '_总结.txt'
    txt_filepath = os.path.join(app.config['PROCESSED_FOLDER'], txt_filename)

    with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
        txt_file.write(translated_content)


# 构建已上传文件图谱
@app.route('/build_knowledge_graph', methods=['POST'])
def build_knowledge_graph():
    # 初始化图知识库（图形数据库需要更改URL地址与账号密码）
    graph_base = module.GraphBase()
    # 构建知识图谱（先翻译再构建）
    result = graph_base.build_knowledge_graph(folder_path=app.config['UPLOAD_FOLDER'])
    if result is True:
        return jsonify({"summary_success": True})
    else:
        return jsonify({"summary_success": False})


# 删除已上传文件图谱
@app.route('/delete_knowledge_graph', methods=['POST'])
def delete_knowledge_graph():
    graph_base = module.GraphBase()
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

    query = module.QueryKnowledgeBase(chroma_path=chroma_path, file_num=file_num)

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
    load_uploaded_files()
    app.run(host='0.0.0.0', debug=False, port=4398)
