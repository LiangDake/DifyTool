def upload_files():
    global uploaded_files
    if 'files[]' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files[]')

    for uploaded_file in files:
        if uploaded_file.filename == '':
            continue

        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
        uploaded_file.save(filepath)

        # 处理 .eml 文件
        if uploaded_file.filename.endswith('.eml'):
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
                    # 解码附件文件名
                    filename = email.header.decode_header(filename)[0][0]
                    if isinstance(filename, bytes):
                        filename = filename.decode()

                    # 保存附件到指定文件夹
                    attachment_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(attachment_path, "wb") as f:
                        f.write(part.get_payload(decode=True))
                    if filename not in uploaded_files:
                        uploaded_files.append(filename)

        # 保存 .eml 文件本身
        if uploaded_file.filename not in uploaded_files:
            uploaded_files.append(uploaded_file.filename)

    return redirect(url_for('index'))