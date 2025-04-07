import os
import re
import subprocess

def md_to_latex(md_file, tex_file, pdf_file):
    output_dir = os.path.dirname(tex_file) 
    os.makedirs(output_dir, exist_ok=True)  

    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.readlines()

    # LaTeX preamble
    tex_content = [
        "\\documentclass{article}\n",
        "\\usepackage{arxiv}\n",
        "\\usepackage{graphicx}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage{hyperref}\n",
        "\\usepackage{multicol}\n",
        "\\usepackage[numbers]{natbib}\n",
        "\\begin{document}\n"
    ]

    title = None
    body_content = []
    references_section = False
    references = []
    citation_map = {} 
    figure_counter = 1

    for i, line in enumerate(md_content):
        line = line.strip()
        line = line.replace("&", "\\&")  # Escape '&' for LaTeX

        # Convert Markdown bold/italic to LaTeX
        line = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", line)
        line = re.sub(r"\*(.*?)\*", r"\\textit{\1}", line)

        if line.startswith("# "):
            title = line[2:]

        elif line.startswith("## "):
            if line.strip() == "## References":
                references_section = True
                body_content.append("\\begin{thebibliography}{99}\n")
            else:
                body_content.append(f"\\section*{{{line[3:]}}}\n")

        elif line.startswith("### "):
            body_content.append(f"\\subsection*{{{line[4:]}}}\n")

        elif references_section:
            match = re.match(r"\[(\d+)\] (.+)", line)
            if match:
                ref_id, ref_text = match.groups()
                citation_map[ref_id] = ref_id  
                references.append(f"\\bibitem{{{ref_id}}} {ref_text}\n")

        elif re.match(r"!\[.*\]\((.*?)\)", line):
            img_match = re.match(r"!\[.*\]\((.*?)\)", line)
            img_path = img_match.group(1)
            img_path = img_path.replace("\\", "/") 

            caption = ""
            if i + 1 < len(md_content) and "**Figure Caption:**" in md_content[i + 1]:
                caption = re.sub(r"\*\*Figure Caption:\*\*\s*", "", md_content[i + 1].strip())

            body_content.append("\\begin{figure}[h]\n\\centering\n")
            body_content.append(f"\\includegraphics[width=0.9\\linewidth]{{{img_path}}}\n")
            if caption:
                body_content.append(f"\\caption{{Figure {figure_counter}: {caption}}}\n")
                figure_counter += 1
            body_content.append("\\end{figure}\n\n")

        else:
            # Convert citations outside Reference
            line = re.sub(r"\[(\d+)\]", lambda m: f"\\cite{{{m.group(1)}}}", line)
            body_content.append(line + '\n')

    if references:
        body_content.append("\n".join(references))
        body_content.append("\\end{thebibliography}\n")

    # Add title and author
    if title:
        tex_content.append(f"\\title{{{title}}}\n")
    tex_content.append("\\author{Artificial Intelligence}\n")
    tex_content.append("\\date{\\today}\n")
    tex_content.append("\\maketitle\n")
    tex_content.append("\\noindent\n")
    tex_content.append("\\twocolumn\n")

    tex_content.extend(body_content)
    tex_content.append("\n\\end{document}\n")

    with open(tex_file, 'w', encoding='utf-8') as f:
        f.writelines(tex_content)

    print(f"LaTeX file saved as {tex_file}")

    pdf_output_path = os.path.join(output_dir, os.path.basename(pdf_file))  # Set PDF output path

    # Run pdflatex twice for citations
    subprocess.run(["pdflatex", "-output-directory", output_dir, tex_file])
    subprocess.run(["pdflatex", "-output-directory", output_dir, tex_file])

    print(f"PDF generated: {pdf_output_path}")
