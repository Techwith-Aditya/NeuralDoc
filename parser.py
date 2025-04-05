import fitz
import json
import os
import nest_asyncio
import re
import sys
import torch

from dotenv import load_dotenv
from llama_parse import LlamaParse
from sentence_transformers import SentenceTransformer

# for async operations in Jupyter environments
nest_asyncio.apply()

class LlamaPDFParser:
    def __init__(self, pdf_path, output_md_path, output_json_path, image_output_folder):
        """PDF parser with markdown conversion, image extraction, and embedding capabilities"""
        load_dotenv()
        self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("Missing LLAMA_CLOUD_API_KEY in .env")

        # Initialize embedding model with fallback
        self.embedding_model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

        self.pdf_path = pdf_path
        self.output_md_path = output_md_path
        self.output_json_path = output_json_path
        self.image_output_path = image_output_folder
        self.documents, self.images_with_caption = self._parse_pdf_to_markdown()

    def _parse_pdf_to_markdown(self):
        """Core PDF parsing using LlamaParse with markdown output"""
        parser = LlamaParse(
            result_type="markdown",
            premium_mode=True,  
        )
        try:
            documents = parser.load_data(self.pdf_path)
            markdown_content = "\n\n".join([doc.text for doc in documents])
            
            os.makedirs(os.path.dirname(self.output_md_path), exist_ok=True)
            with open(self.output_md_path, "w", encoding="utf-8") as md_file:
                md_file.write(markdown_content)

            # Extract and append images with captions
            images_with_caption = self._extract_images_with_captions()
            with open(self.output_md_path, "a", encoding="utf-8") as md_file:
                for img in images_with_caption:
                    md_file.write(f"\n![Image]({img['metadata']['image']})\n**Caption:** {img['metadata']['caption']}\n\n")            
            
            return markdown_content, images_with_caption
        except Exception as e:
            raise ValueError(f"PDF parsing failed: {e}")

    def _extract_images_with_captions(self):
        """Extract images with contextual captions using spatial analysis"""
        os.makedirs(self.image_output_path, exist_ok=True)
        doc = fitz.open(self.pdf_path)
        image_docs = []

        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("blocks")
            image_docs.extend(self.parse_all_images(self.image_output_path, page, page_num + 1, text_blocks))

        return image_docs

    def parse_all_images(self, filename, page, pagenum, text_blocks):
        """Image extraction with spatial context analysis"""
        image_docs = []
        page_rect = page.rect

        for image_info in page.get_image_info(xrefs=True):
            xref = image_info['xref']
            if xref == 0:
                continue

            # Filter small images
            img_bbox = fitz.Rect(image_info['bbox'])
            if img_bbox.width < page_rect.width/20 or img_bbox.height < page_rect.height/20:
                continue

            extracted_image = page.parent.extract_image(xref)
            imgrefpath = os.path.join(os.getcwd(), f"{filename}")
            os.makedirs(imgrefpath, exist_ok=True)
            image_path = os.path.join(imgrefpath, f"image{xref}-page{pagenum}.png")
            with open(image_path, "wb") as img_file:
                img_file.write(extracted_image["image"])

            # Generate contextual caption
            before_text, after_text = self.extract_text_around_item(text_blocks, img_bbox, page.rect.height)
            caption = f"{before_text} {after_text}".replace("\n", " ").strip()

            image_docs.append({
                "text": "This is an image with the caption: " + caption,
                "metadata": {
                    "source": f"{filename}-page{pagenum}-image{xref}",
                    "image": image_path,
                    "caption": caption,
                    "type": "image",
                    "page_num": pagenum
                }
            })

        return image_docs

    def _parse_markdown_to_json(self, md_content):
        """Convert markdown hierarchy to structured JSON"""
        hierarchy = {}
        current_levels = []

        for line in md_content.splitlines():
            # Process headings and build hierarchy
            if heading_match := re.match(r"^(#+) (.+)", line):
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                current_levels = current_levels[:level-1]
                current_levels.append(heading_text)

                metadata = {
                    "main title": current_levels[0] if len(current_levels) > 0 else "",
                    "section title": current_levels[1] if len(current_levels) > 1 else "",
                    "sub heading": current_levels[2] if len(current_levels) > 2 else ""
                }

                current_node = hierarchy
                for lvl in current_levels[:-1]:
                    current_node = current_node.setdefault(lvl, {"content": "", "subheadings": {}})["subheadings"]
                current_node[current_levels[-1]] = {"content": "", "metadata": metadata, "subheadings": {}}

            # Accumulate content for current heading
            elif current_levels:
                current_node = hierarchy
                for lvl in current_levels[:-1]:
                    current_node = current_node[lvl]["subheadings"]
                current_node[current_levels[-1]]["content"] += line.strip() + "\n"

        return hierarchy

    def generate_embeddings(self):
        """Generate embeddings for document hierarchy using E5-large model"""
        nodes = self.get_text_page_nodes()

        for node in nodes:
            # Encodes main and section headings separately
            node['embeddings-Main-Headding'] = self.embedding_model.encode(
                node['metadata'].get('main title', '')
            )
            node['embeddings-Section-Headding'] = self.embedding_model.encode(
                node['metadata'].get('section title', '')
            )

        return nodes

def main():
    """CLI entry point for PDF processing"""
    if len(sys.argv) < 2:
        print("Usage: python parser.py <pdf_file>")
        sys.exit(1)

    parser = LlamaPDFParser(
        sys.argv[1],
        "output/cnn1.md",
        "output/cnn1.json",
        "output/images"
    )
    
    # Debug output for parsed document structure
    hierarchy = parser._parse_markdown_to_json(parser.documents)
    print(hierarchy)

if __name__ == "__main__":
    main()
