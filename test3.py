import email
import os

# 本地邮件文件路径
eml_file_path = "/Users/liangdake/Downloads/工作/人工智能实习/Studentsafe Policy - Record of Cover - Student ID 300666449.eml"

# 定义保存附件的文件夹路径
folder_path = "/Users/liangdake/Downloads/工作/人工智能实习/email"
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

# 读取邮件内容
with open(eml_file_path, 'r') as file:
    msg = email.message_from_file(file)

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
        # 如果文件名包含中文，需要进行解码
        filename = email.header.decode_header(filename)[0][0]
        if isinstance(filename, bytes):
            filename = filename.decode()

        # 保存附件到指定文件夹
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "wb") as f:
            f.write(part.get_payload(decode=True))

        print(f"Attachment {filename} saved to {file_path}")
