"""
export_tools.py

本脚本整合了原 export_visualization_to_pdf.py、export_visualization_to_word.py 等导出相关功能。
每个功能以函数形式提供，便于统一调用和维护。
如需扩展更多导出功能，可在本脚本中补充。

合并来源：
- export_visualization_to_pdf.py
- export_visualization_to_word.py

用法示例：
    from export_tools import export_to_pdf, export_to_word
    export_to_pdf(...)
    export_to_word(...)
"""

def export_to_pdf(input_dir, output_path):
    """
    将input_dir下所有图片导出为PDF报告。
    支持多级目录，按文件名排序。
    """
    import os
    import img2pdf
    img_list = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png')):
                img_list.append(os.path.abspath(os.path.join(root, f)))
    img_list = sorted(img_list)
    if img_list:
        with open(output_path, "wb") as f:
            f.write(img2pdf.convert(img_list))
        print(f"已生成PDF: {output_path}")
    else:
        print("未找到可导出的图片文件")

def export_to_word(input_dir, output_path):
    """
    将input_dir下所有样本/epoch目录下图片导出为Word报告。
    支持多级目录，按目录和文件名排序。
    """
    import os
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    doc.add_heading('可视化导出报告', 0)
    for root, dirs, files in os.walk(input_dir):
        if files:
            doc.add_heading(os.path.relpath(root, input_dir), level=1)
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.abspath(os.path.join(root, f))
                    doc.add_paragraph(f)
                    doc.add_picture(img_path, width=Inches(2.0))
    doc.save(output_path)
    print(f"已生成Word: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeeplabV3+ 可视化导出工具集")
    subparsers = parser.add_subparsers(dest='command')

    parser_pdf = subparsers.add_parser('pdf', help='导出为PDF')
    parser_pdf.add_argument('--input_dir', type=str, required=True)
    parser_pdf.add_argument('--output', type=str, required=True)

    parser_word = subparsers.add_parser('word', help='导出为Word')
    parser_word.add_argument('--input_dir', type=str, required=True)
    parser_word.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    if args.command == 'pdf':
        export_to_pdf(args.input_dir, args.output)
    elif args.command == 'word':
        export_to_word(args.input_dir, args.output)
    else:
        parser.print_help()
