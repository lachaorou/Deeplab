import os
import markdown2
from fpdf import FPDF

# 将by_sample下所有compare.md合并为一个大md，并尝试导出为pdf
# 若pdf导出失败，保留合并后的md

def collect_md_files(by_sample_dir):
    md_files = []
    for d in os.listdir(by_sample_dir):
        sub = os.path.join(by_sample_dir, d)
        if os.path.isdir(sub):
            md_path = os.path.join(sub, 'compare.md')
            if os.path.exists(md_path):
                md_files.append(md_path)
    return md_files

def merge_md(md_files, out_md):
    with open(out_md, 'w', encoding='utf-8') as fout:
        for md in md_files:
            with open(md, 'r', encoding='utf-8') as fin:
                fout.write(fin.read())
                fout.write('\n\n---\n\n')
    print(f"[合并] 已生成: {out_md}")

def md_to_pdf(md_path, pdf_path):
    try:
        html = markdown2.markdown_path(md_path)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        # 只保留文本内容，图片路径不渲染
        import re
        text = re.sub(r'!\[.*?\]\(.*?\)', '[图片]', html)
        for line in text.split('\n'):
            pdf.cell(0, 10, line, ln=1)
        pdf.output(pdf_path)
        print(f"[PDF] 已生成: {pdf_path}")
        return True
    except Exception as e:
        print(f"[PDF] 生成失败: {e}")
        return False

if __name__ == '__main__':
    by_sample_dir = os.path.join(os.path.dirname(__file__), '..', 'Inspect', 'VisualizeCompare', 'by_sample')
    out_md = os.path.join(by_sample_dir, 'all_compare.md')
    out_pdf = os.path.join(by_sample_dir, 'all_compare.pdf')
    md_files = collect_md_files(by_sample_dir)
    merge_md(md_files, out_md)
    md_to_pdf(out_md, out_pdf)
    print('[完成] 可视化对比已合并并尝试导出PDF。')
